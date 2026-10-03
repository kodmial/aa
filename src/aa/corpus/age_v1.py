"""Pure-Python age v1 file encryption for the X25519 recipient type.

This module implements just enough of the ``age-encryption.org/v1`` format
to satisfy issue #24 without requiring the external ``age`` binary:

- X25519 identities (``AGE-SECRET-KEY-...``) and recipients (``age1...``);
- single- or multi-recipient X25519 headers with HMAC-SHA-256 header MAC;
- STREAM ChaCha20-Poly1305 payload encryption in 64 KiB chunks.

The implementation follows the C2SP age specification and is wire-compatible
with the reference Go/Rust ``age`` tools for X25519 recipients. It does not
implement scrypt, SSH, plugin, armor, or post-quantum recipient types.

Only ``cryptography`` (X25519, ChaCha20-Poly1305, HKDF) plus the standard
library is used. No plaintext or key material is ever logged from here.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import struct

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

VERSION_LINE = b"age-encryption.org/v1\n"
CHUNK_SIZE = 64 * 1024
FILE_KEY_SIZE = 16
NONCE_SIZE = 16
PAYLOAD_KEY_INFO = b"payload"
HEADER_KEY_INFO = b"header"
X25519_INFO = b"age-encryption.org/v1/X25519"
RECIPIENT_HRP = "age"
IDENTITY_HRP = "AGE-SECRET-KEY-"

_ZERO_NONCE = b"\x00" * 12


class AgeError(ValueError):
    """Raised when an age payload fails validation (fails closed)."""


# ---------------------------------------------------------------------------
# Base64 without padding (RFC 4648 section 4, raw/unpadded).
# ---------------------------------------------------------------------------


def b64encode_nopad(data: bytes) -> str:
    """Encode bytes as canonical unpadded base64."""
    return base64.b64encode(data).decode("ascii").rstrip("=")


def b64decode_nopad(text: str) -> bytes:
    """Decode unpadded base64, rejecting padding and non-canonical input."""
    if not text:
        return b""
    if "=" in text:
        raise AgeError("base64 input must not contain padding")
    _B64_ALPHABET = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/")
    if any(c not in _B64_ALPHABET for c in text):
        raise AgeError("base64 input contains invalid characters")
    # Canonical form: re-encoding the decoded bytes must reproduce the input.
    padded = text + "=" * (-len(text) % 4)
    try:
        raw = base64.b64decode(padded.encode("ascii"), validate=True)
    except Exception as exc:
        raise AgeError(f"invalid base64 input: {exc}") from exc
    if b64encode_nopad(raw) != text:
        raise AgeError("non-canonical base64 input")
    return raw


# ---------------------------------------------------------------------------
# Bech32 (BIP173, without length limits on the data part).
# ---------------------------------------------------------------------------

_BECH32_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"


def _bech32_polymod(values: list[int]) -> int:
    generator = [0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3]
    chk = 1
    for value in values:
        top = chk >> 25
        chk = ((chk & 0x1FFFFFF) << 5) ^ value
        for i in range(5):
            chk ^= generator[i] if ((top >> i) & 1) else 0
    return chk


def _bech32_hrp_expand(hrp: str) -> list[int]:
    return [ord(x) >> 5 for x in hrp] + [0] + [ord(x) & 31 for x in hrp]


def _bech32_verify_checksum(hrp: str, data: list[int]) -> bool:
    return _bech32_polymod(_bech32_hrp_expand(hrp) + data) == 1


def _bech32_create_checksum(hrp: str, data: list[int]) -> list[int]:
    values = _bech32_hrp_expand(hrp) + data
    polymod = _bech32_polymod(values + [0, 0, 0, 0, 0, 0]) ^ 1
    return [(polymod >> 5 * (5 - i)) & 31 for i in range(6)]


def _convertbits(data: bytes, frombits: int, tobits: int, pad: bool) -> list[int]:
    acc = 0
    bits = 0
    result: list[int] = []
    maxv = (1 << tobits) - 1
    for byte in data:
        acc = (acc << frombits) | byte
        bits += frombits
        while bits >= tobits:
            bits -= tobits
            result.append((acc >> bits) & maxv)
    if pad:
        if bits:
            result.append((acc << (tobits - bits)) & maxv)
    elif bits >= frombits or ((acc << (tobits - bits)) & maxv):
        raise AgeError("invalid bech32 padding")
    return result


def _convertbits_to_bytes(values: list[int], frombits: int, tobits: int) -> bytes:
    acc = 0
    bits = 0
    result = bytearray()
    maxv = (1 << tobits) - 1
    for value in values:
        if value < 0 or value >> frombits:
            raise AgeError("invalid bech32 data value")
        acc = (acc << frombits) | value
        bits += frombits
        while bits >= tobits:
            bits -= tobits
            result.append((acc >> bits) & maxv)
    # Per BIP173, leftover bits must be zero and fewer than frombits.
    if bits >= frombits or ((acc << (tobits - bits)) & maxv):
        raise AgeError("invalid bech32 padding")
    return bytes(result)


def _bech32_encode(hrp: str, raw: bytes, *, upper: bool = False) -> str:
    data = _convertbits(raw, 8, 5, True)
    combined = data + _bech32_create_checksum(hrp.lower(), data)
    encoded = hrp + "1" + "".join(_BECH32_CHARSET[d] for d in combined)
    return encoded.upper() if upper else encoded.lower()


def _bech32_decode(text: str, *, expected_hrp: str) -> bytes:
    if text.lower() != text and text.upper() != text:
        raise AgeError("mixed-case bech32 string")
    lowered = text.lower()
    if "1" not in lowered:
        raise AgeError("missing bech32 separator")
    pos = lowered.rfind("1")
    hrp = lowered[:pos]
    if hrp != expected_hrp.lower():
        raise AgeError(f"unexpected bech32 HRP: {hrp!r}")
    data_part = lowered[pos + 1 :]
    if len(data_part) < 6:
        raise AgeError("bech32 data part too short")
    try:
        values = [_BECH32_CHARSET.index(c) for c in data_part]
    except ValueError as exc:
        raise AgeError(f"invalid bech32 character: {exc}") from exc
    if not _bech32_verify_checksum(hrp, values):
        raise AgeError("invalid bech32 checksum")
    return _convertbits_to_bytes(values[:-6], 5, 8)


# ---------------------------------------------------------------------------
# Key material.
# ---------------------------------------------------------------------------


def generate_identity() -> tuple[str, str]:
    """Generate a fresh X25519 identity and its recipient.

    Returns ``(identity, recipient)`` where the identity uses the
    ``AGE-SECRET-KEY-`` HRP (uppercase) and the recipient uses ``age1``.
    """
    secret = os.urandom(32)
    private = X25519PrivateKey.from_private_bytes(secret)
    public = private.public_key().public_bytes_raw()
    identity = _bech32_encode(IDENTITY_HRP, secret, upper=True)
    recipient = _bech32_encode(RECIPIENT_HRP, public, upper=False)
    return identity, recipient


def parse_identity(text: str) -> bytes:
    """Parse an ``AGE-SECRET-KEY-...`` identity into 32 secret bytes."""
    cleaned = text.strip()
    raw = _bech32_decode(cleaned, expected_hrp=IDENTITY_HRP)
    if len(raw) != 32:
        raise AgeError("invalid identity length")
    return raw


def parse_recipient(text: str) -> bytes:
    """Parse an ``age1...`` recipient into 32 public-key bytes."""
    cleaned = text.strip()
    raw = _bech32_decode(cleaned, expected_hrp=RECIPIENT_HRP)
    if len(raw) != 32:
        raise AgeError("invalid recipient length")
    # Validate that the bytes are a usable X25519 public key.
    try:
        X25519PublicKey.from_public_bytes(raw)
    except Exception as exc:
        raise AgeError(f"invalid X25519 recipient: {exc}") from exc
    return raw


def _hkdf(ikm: bytes, salt: bytes, info: bytes) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=salt, info=info).derive(ikm)


def _wrap_file_key(file_key: bytes, recipient_pub: bytes) -> tuple[bytes, bytes]:
    """Wrap ``file_key`` for one recipient; returns (share, body)."""
    while True:
        ephemeral_secret = os.urandom(32)
        ephemeral_private = X25519PrivateKey.from_private_bytes(ephemeral_secret)
        ephemeral_share = ephemeral_private.public_key().public_bytes_raw()
        recipient_key = X25519PublicKey.from_public_bytes(recipient_pub)
        shared = ephemeral_private.exchange(recipient_key)
        if shared != b"\x00" * 32:
            break
    wrap_key = _hkdf(shared, ephemeral_share + recipient_pub, X25519_INFO)
    body = ChaCha20Poly1305(wrap_key).encrypt(_ZERO_NONCE, file_key, None)
    return ephemeral_share, body


def _unwrap_file_key(
    *,
    identity_secret: bytes,
    ephemeral_share: bytes,
    recipient_pub: bytes | None,
    body: bytes,
    known_recipients: list[bytes] | None = None,
) -> bytes:
    """Unwrap a stanza body; tries derived recipient when not supplied."""
    private = X25519PrivateKey.from_private_bytes(identity_secret)
    try:
        share_key = X25519PublicKey.from_public_bytes(ephemeral_share)
    except Exception as exc:
        raise AgeError(f"invalid ephemeral share: {exc}") from exc
    shared = private.exchange(share_key)
    if shared == b"\x00" * 32:
        raise AgeError("invalid shared secret")
    candidates: list[bytes] = []
    derived = private.public_key().public_bytes_raw()
    candidates.append(derived)
    if recipient_pub is not None:
        candidates.append(recipient_pub)
    if known_recipients:
        candidates.extend(known_recipients)
    last_error: Exception | None = None
    for candidate in candidates:
        wrap_key = _hkdf(shared, ephemeral_share + candidate, X25519_INFO)
        try:
            file_key = ChaCha20Poly1305(wrap_key).decrypt(_ZERO_NONCE, body, None)
        except InvalidTag as exc:
            last_error = exc
            continue
        if len(file_key) != FILE_KEY_SIZE:
            raise AgeError("invalid file key length")
        return file_key
    raise AgeError(f"file key unwrap failed: {last_error}")


def _encode_stanza_body(body: bytes) -> str:
    encoded = b64encode_nopad(body)
    lines = [encoded[i : i + 64] for i in range(0, len(encoded), 64)]
    # The body MUST end with a line shorter than 64 characters (MAY be empty).
    if lines and len(lines[-1]) == 64:
        lines.append("")
    if not lines:
        lines = [""]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# File encryption.
# ---------------------------------------------------------------------------


def encrypt_bytes(plaintext: bytes, recipients: list[str]) -> bytes:
    """Encrypt ``plaintext`` to one or more ``age1...`` recipients."""
    if not recipients:
        raise AgeError("at least one recipient is required")
    recipient_pubs = [parse_recipient(item) for item in recipients]
    file_key = os.urandom(FILE_KEY_SIZE)

    header = bytearray(VERSION_LINE)
    for pub in recipient_pubs:
        share, body = _wrap_file_key(file_key, pub)
        header += f"-> X25519 {b64encode_nopad(share)}\n".encode("ascii")
        header += _encode_stanza_body(body).encode("ascii")

    header_key = _hkdf(file_key, b"", HEADER_KEY_INFO)
    # Per the spec the MAC covers the header up to and including the "---"
    # mark, excluding the space that follows it in the MAC line.
    prefix = bytes(header) + b"---"
    mac = hmac.new(header_key, prefix, hashlib.sha256).digest()
    header += b"--- " + b64encode_nopad(mac).encode("ascii") + b"\n"

    nonce = os.urandom(NONCE_SIZE)
    payload_key = _hkdf(file_key, nonce, PAYLOAD_KEY_INFO)
    aead = ChaCha20Poly1305(payload_key)

    out = bytearray(bytes(header) + nonce)
    if len(plaintext) == 0:
        chunk_nonce = b"\x00" * 11 + b"\x01"
        out += aead.encrypt(chunk_nonce, b"", None)
        return bytes(out)
    offset = 0
    counter = 0
    while offset < len(plaintext):
        chunk = plaintext[offset : offset + CHUNK_SIZE]
        offset += CHUNK_SIZE
        last = b"\x01" if offset >= len(plaintext) else b"\x00"
        chunk_nonce = b"\x00\x00\x00" + struct.pack(">Q", counter) + last
        out += aead.encrypt(chunk_nonce, chunk, None)
        counter += 1
    return bytes(out)


def _split_header(data: bytes) -> tuple[list[tuple[list[str], bytes]], bytes, int]:
    """Split the age header into stanzas; returns (stanzas, rest, hdr_end)."""
    if not data.startswith(VERSION_LINE):
        raise AgeError("unsupported age version line")
    lines = data.split(b"\n")
    # lines[0] is the version line without its newline.
    stanzas: list[tuple[list[str], bytes]] = []
    index = 1
    current_args: list[str] | None = None
    current_body = ""
    # Number of bytes consumed including newlines.
    consumed = len(VERSION_LINE)
    while index < len(lines):
        line = lines[index]
        line_len_with_nl = len(line) + 1
        if line.startswith(b"-> "):
            if current_args is not None:
                raise AgeError("malformed stanza: missing body terminator")
            try:
                current_args = line[3:].decode("ascii").split(" ")
            except UnicodeDecodeError as exc:
                raise AgeError(f"invalid stanza arguments: {exc}") from exc
            if len(current_args) < 1 or any(not a for a in current_args):
                raise AgeError("malformed stanza arguments")
            current_body = ""
            consumed += line_len_with_nl
            index += 1
            continue
        if line.startswith(b"--- "):
            if current_args is not None:
                # A stanza body ends with a short line; a MAC line also ends
                # the stanza. The accumulated body is complete.
                body = b64decode_nopad(current_body) if current_body else b""
                # Re-validate wrapping: body must have been wrapped at 64 cols.
                stanzas.append((current_args, body))
                current_args = None
                current_body = ""
            try:
                mac_text = line[4:].decode("ascii")
            except UnicodeDecodeError as exc:
                raise AgeError(f"invalid MAC line: {exc}") from exc
            mac = b64decode_nopad(mac_text)
            if len(mac) != 32:
                raise AgeError("invalid header MAC length")
            # Header ends after this line's newline.
            hdr_end = consumed + line_len_with_nl
            rest = data[hdr_end:]
            return stanzas, rest, hdr_end
        # Body line of the current stanza.
        if current_args is None:
            raise AgeError("malformed age header")
        try:
            text = line.decode("ascii")
        except UnicodeDecodeError as exc:
            raise AgeError(f"invalid stanza body: {exc}") from exc
        if len(line) > 64:
            raise AgeError("stanza body line exceeds 64 columns")
        if len(line) == 64:
            current_body += text
        else:
            current_body += text
            body = b64decode_nopad(current_body) if current_body else b""
            stanzas.append((current_args, body))
            current_args = None
            current_body = ""
        consumed += line_len_with_nl
        index += 1
    raise AgeError("age header is missing its MAC line")


def decrypt_bytes(ciphertext: bytes, identities: list[str]) -> bytes:
    """Decrypt age ``ciphertext`` with one or more ``AGE-SECRET-KEY-...``."""
    if not identities:
        raise AgeError("at least one identity is required")
    secrets = [parse_identity(item) for item in identities]
    stanzas, rest, _ = _split_header(ciphertext)
    if not stanzas:
        raise AgeError("age header has no recipient stanzas")

    # Collect candidate file keys from X25519 stanzas.
    candidates: list[bytes] = []
    for args, body in stanzas:
        if not args or args[0] != "X25519":
            continue
        if len(args) != 2:
            raise AgeError("malformed X25519 stanza")
        share = b64decode_nopad(args[1])
        if len(share) != 32:
            raise AgeError("invalid X25519 ephemeral share length")
        if len(body) != 32:
            raise AgeError("invalid X25519 stanza body length")
        for secret in secrets:
            private = X25519PrivateKey.from_private_bytes(secret)
            derived = private.public_key().public_bytes_raw()
            try:
                file_key = _unwrap_file_key(
                    identity_secret=secret,
                    ephemeral_share=share,
                    recipient_pub=derived,
                    body=body,
                )
            except AgeError:
                continue
            candidates.append(file_key)
            break
    if not candidates:
        raise AgeError("no matching X25519 stanza for the provided identities")

    last_error: Exception | None = None
    for file_key in candidates:
        try:
            return _decrypt_with_file_key(ciphertext, rest, file_key)
        except AgeError as exc:
            last_error = exc
            continue
    raise AgeError(f"age decryption failed: {last_error}")


def _decrypt_with_file_key(ciphertext: bytes, rest: bytes, file_key: bytes) -> bytes:
    # Verify header MAC before touching the payload.
    header_end = len(ciphertext) - len(rest)
    header_prefix_end = ciphertext.rfind(b"\n--- ")
    if header_end <= 0 or header_prefix_end < 0:
        raise AgeError("malformed age header")
    mac_line = ciphertext[header_prefix_end + len(b"\n--- ") : header_end].rstrip(b"\n")
    try:
        mac = b64decode_nopad(mac_line.decode("ascii"))
    except UnicodeDecodeError as exc:
        raise AgeError(f"invalid MAC line: {exc}") from exc
    header_key = _hkdf(file_key, b"", HEADER_KEY_INFO)
    # MAC covers the header up to and including "---" (no trailing space).
    expected = hmac.new(
        header_key, ciphertext[: header_prefix_end + 1] + b"---", hashlib.sha256
    ).digest()
    if not hmac.compare_digest(mac, expected):
        raise AgeError("invalid header MAC")
    if len(rest) < NONCE_SIZE + 16:
        raise AgeError("age payload is truncated")
    nonce = rest[:NONCE_SIZE]
    body = rest[NONCE_SIZE:]
    payload_key = _hkdf(file_key, nonce, PAYLOAD_KEY_INFO)
    aead = ChaCha20Poly1305(payload_key)

    # Each encrypted chunk is plaintext (<=64KiB) + 16-byte tag.
    out = bytearray()
    offset = 0
    counter = 0
    found_final = False
    while offset < len(body):
        remaining = len(body) - offset
        # The final chunk is the one whose decrypted last-flag byte is 0x01;
        # both final and non-final chunks carry a 16-byte tag, so try the
        # longest plausible non-final chunk first, then the remainder as final.
        is_last_piece = remaining <= CHUNK_SIZE + 16
        if is_last_piece:
            chunk_ct = body[offset:]
            chunk_nonce = b"\x00\x00\x00" + struct.pack(">Q", counter) + b"\x01"
            try:
                out += aead.decrypt(chunk_nonce, chunk_ct, None)
            except InvalidTag as exc:
                raise AgeError(f"payload chunk {counter} failed authentication") from exc
            found_final = True
            offset = len(body)
        else:
            chunk_ct = body[offset : offset + CHUNK_SIZE + 16]
            chunk_nonce = b"\x00\x00\x00" + struct.pack(">Q", counter) + b"\x00"
            try:
                out += aead.decrypt(chunk_nonce, chunk_ct, None)
            except InvalidTag as exc:
                raise AgeError(f"payload chunk {counter} failed authentication") from exc
            offset += CHUNK_SIZE + 16
        counter += 1
    if not found_final:
        raise AgeError("age payload is missing its final chunk")
    return bytes(out)
