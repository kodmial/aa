#!/usr/bin/env bash
# Container Git trust bootstrap (kodmial/aa#435).
#
# actions/checkout runs `git config --global --add safe.directory ...` inside
# a temporary overridden HOME, so ordinary later steps in a container job
# still fail with `fatal: detected dubious ownership in repository at ...`
# (exit 128) whenever the mounted workspace UID differs from the effective
# process UID. This script restores Git trust for EXACTLY the checked-out
# workspace in the trusted config that later ordinary steps actually read:
#
#   - validates GITHUB_WORKSPACE (nonempty, absolute, canonical, an actual
#     git repository root);
#   - adds `safe.directory` with that exact canonical path to the job's
#     effective persistent trusted Git config (global config resolved the
#     same way future `git` processes resolve it; system config only if the
#     global location is unavailable);
#   - refuses wildcards, static runner-specific paths, permission bypasses
#     (no chown/chmod, no ownership-validation disabling) and never amends
#     trusted config for anything outside this exact workspace;
#   - fail-closed: exits clearly when no writable trusted config exists;
#   - verifies afterwards that ordinary git sees the exact workspace
#     toplevel, a full HEAD SHA and a readable working-tree status;
#   - is idempotent: re-invocation never duplicates the exact entry.

set -euo pipefail

die() {
  echo "::error::$*" >&2
  exit 1
}

notice() {
  echo "$*"
}

# --- 1. Validate the exact checked-out workspace ----------------------------

workspace_env="${GITHUB_WORKSPACE:-}"
[[ -n "$workspace_env" ]] \
  || die "GITHUB_WORKSPACE is empty; refusing to bootstrap Git trust for an unknown directory."
[[ "$workspace_env" == /* ]] \
  || die "GITHUB_WORKSPACE is not an absolute path: '$workspace_env'."
if printf '%s' "$workspace_env" | grep -q '\*'; then
  die "GITHUB_WORKSPACE contains wildcard characters; only an exact literal directory may be trusted."
fi
if ! workspace="$(cd -- "$workspace_env" && pwd -P)"; then
  die "GITHUB_WORKSPACE is not a usable checked-out directory: '$workspace_env'."
fi
if [[ "$workspace" != "$workspace_env" ]]; then
  die "GITHUB_WORKSPACE is not a canonical absolute path: '$workspace_env' vs '$workspace'."
fi
if [[ ! -e "$workspace/.git" ]]; then
  die "GITHUB_WORKSPACE is not a git repository root (no .git entry): '$workspace'."
fi
if [[ ! -d "$workspace" || ! -r "$workspace" || ! -x "$workspace" ]]; then
  die "GITHUB_WORKSPACE is not a readable searchable directory: '$workspace'."
fi

# --- 2. Choose the trusted config location later ordinary steps will read --
#
# Mirror git's own resolution so the written entry is exactly what the later
# shell steps read: a GIT_CONFIG_GLOBAL override first, then the job's
# persistent HOME, then (only if no global location is usable) the system
# config. When HOME is unset/absolute-unusable, never silently redirect to
# another user's home; use the system config instead.

run_git_config() {
  # `git config` must not run with the dubious repository as its current
  # directory; a neutral cwd keeps repo-associated setup out of this step.
  ( cd / && git config "$@" )
}

if [[ -n "${GIT_CONFIG_GLOBAL:-}" ]]; then
  global_dir="$(dirname -- "$GIT_CONFIG_GLOBAL")"
  if [[ ! -d "$global_dir" || ! -w "$global_dir" ]]; then
    die "GIT_CONFIG_GLOBAL is set but its parent directory is not writable: refusing to continue untrusted."
  fi
  scope_flag="--global"
  get_flag="--global"
  config_origin="$GIT_CONFIG_GLOBAL"
elif [[ -n "${HOME:-}" && -d "$HOME" && -w "$HOME" ]]; then
  scope_flag="--global"
  get_flag="--global"
  config_origin="${HOME}/.gitconfig"
else
  # Global trusted config is unavailable for this job's effective user/HOME:
  # fall back to the system config (still exact-path, protected, read by the
  # same user that later ordinary steps run as).
  system_file="${GIT_CONFIG_SYSTEM:-/etc/gitconfig}"
  system_dir="$(dirname -- "$system_file")"
  if [[ ! -d "$system_dir" || ! -w "$system_dir" ]]; then
    die "No writable trusted git config location is available for this job user (HOME unusable and '$system_file' unwritable). Cannot trust the exact workspace."
  fi
  scope_flag="--system"
  get_flag="--system"
  config_origin="$system_file"
fi

# --- 3. Add exactly that directory, idempotently ----------------------------

existing="$(run_git_config "$scope_flag" --get-all safe.directory 2>/dev/null || true)"
if printf '%s\n' "$existing" | grep -Fxq -- "$workspace"; then
  notice "Git trust already present for exactly: $workspace (config: $config_origin)"
else
  run_git_config "$scope_flag" --add safe.directory "$workspace" \
    || die "Writing safe.directory='$workspace' into $config_origin failed; refusing to continue untrusted."
  notice "Git trust added for exactly: $workspace (config: $config_origin)"
fi

# --- 4. Verify ordinary git reads the trusted workspace ---------------------

# Exact-value proof only: the workspace must appear as the whole safe.directory
# value on at least one config line whose origin file is the chosen target.
effective_origin="$(run_git_config "$get_flag" --show-origin --get-all safe.directory 2>/dev/null \
  | awk -F'\t' -v ws="$workspace" '$2 == ws { print; ok = 1 } END { exit ok ? 0 : 1 }' \
  || true)"
if [[ -z "$effective_origin" ]]; then
  die "safe.directory='$workspace' is not readable back as an exact value from $config_origin; trusted config did not stick."
fi

toplevel="$(git -C "$workspace" rev-parse --show-toplevel)" \
  || die "git rev-parse --show-toplevel failed for the trusted workspace."
if [[ "$toplevel" != "$workspace" ]]; then
  die "Trusted directory '$workspace' is not the actual repository root (git reports '$toplevel')."
fi

head_sha="$(git -C "$workspace" rev-parse HEAD)" \
  || die "git rev-parse HEAD failed for the trusted workspace."
if [[ ! "$head_sha" =~ ^[0-9a-f]{40}$ ]]; then
  die "git rev-parse HEAD did not return a full SHA: '$head_sha'."
fi

if ! git -C "$workspace" status --porcelain >/dev/null; then
  die "git status --porcelain failed for the trusted workspace."
fi

notice "Git trust bootstrap proof: toplevel=$toplevel head=$head_sha"
notice "Trusted config origin (matching lines): $effective_origin"
notice "Container git safe.directory bootstrap PASS (exact workspace only)."
