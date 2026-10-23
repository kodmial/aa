"""Container Git safe.directory trust bootstrap contracts (kodmial/aa#435).

Covers the shared bootstrap script (scripts/ensure_container_git_safe_directory.sh)
and every affected AA container workflow job:

- the bootstrap step runs in an ordinary step directly AFTER checkout and
  BEFORE the first other Git-using step;
- the trusted entry is exactly the canonical GITHUB_WORKSPACE (never a
  wildcard, never a static runner path, no chown/chmod/permission bypass);
- repeated invocation is idempotent;
- invalid/missing workspace or an unwritable trusted config fails closed;
- a real differing-owner repository becomes readable for ordinary git
  commands after the bootstrap (the workflow defect from run #39865024944).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ensure_container_git_safe_directory.sh"
WORKFLOWS = ROOT / ".github" / "workflows"
BOOTSTRAP_RUN = "bash scripts/ensure_container_git_safe_directory.sh"
# kodmial/aa#441: the regression workflow pins the single promoted digest
# from docker/aa-runtime.digest (the same source the promotion guard
# enforces); a hardcoded digest here drifted from the promoted image.
VERIFIED_IMAGE_DIGEST = (ROOT / "docker" / "aa-runtime.digest").read_text(encoding="utf-8").strip()

# Every AA container job with confirmed post-checkout Git use (issue #435):
# the two qualification jobs plus the other container jobs that read git
# state after checkout. Hosted-runner-only jobs and qualification gate
# verdicts stay out of scope, and the existing A/B/C/E/F checks are untouched.
AFFECTED_JOBS: dict[str, tuple[str, ...]] = {
    "aa-self-proving-qualification.yml": ("qualify", "qualify-live"),
    "aa-derived-index.yml": ("publish",),
    "aa-runtime.yml": ("runtime",),
    "aa-runtime-image.yml": ("canary",),
    "aa-container-git-trust-regression.yml": ("checkout-bootstrap-verify",),
}

_GIT_USAGE_RE = re.compile(
    r"\bgit\b(?:\s+(?:-C\s+\S+|rev-parse|status|fetch|config|diff|ls-files|merge|commit))"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _job_bounds(lines: list[str], job: str) -> tuple[int, int]:
    """Start index of a job block and the end of its declaration list."""
    job_start = None
    for index, line in enumerate(lines):
        if re.fullmatch(rf"  {re.escape(job)}:\s*(#.*)?", line):
            job_start = index
            break
    assert job_start is not None, f"job {job} not found"
    job_end = len(lines)
    steps_start = None
    scan = job_start + 1
    while scan < len(lines):
        line = lines[scan]
        if line.startswith("    steps:"):
            steps_start = scan
        elif re.fullmatch(r"  \S.*:", line):
            job_end = scan
            break
        scan += 1
    assert steps_start is not None, f"steps not found in job {job}"
    return job_start, min(steps_start, job_end)


def _job_prefix(workflow: str, job: str) -> str:
    """The job block from its declaration up to its steps list."""
    lines = workflow.splitlines()
    job_start, job_steps = _job_bounds(lines, job)
    return "\n".join(lines[job_start:job_steps])


def _job_steps(workflow: str, job: str) -> list[str]:
    """Return the textual block of every named step of one workflow job."""
    lines = workflow.splitlines()
    _, steps_start = _job_bounds(lines, job)
    steps_start += 1
    step_indices: list[int] = []
    job_end = len(lines)
    for index in range(steps_start, job_end):
        line = lines[index]
        if line.startswith("      - name:"):
            step_indices.append(index)
        elif index > steps_start and re.fullmatch(r"  \S.*:", line):
            job_end = index
            break
    assert step_indices, f"no named steps found in job {job}"
    step_indices.append(job_end)
    return [
        "\n".join(lines[start:end])
        for start, end in zip(step_indices[:-1], step_indices[1:], strict=False)
    ]


def _sandbox_env(
    repo: Path,
    home: Path | None = None,
    global_config: Path | None = None,
    system_config: Path | None = None,
) -> dict[str, str]:
    """Minimal deterministic environment mirroring an ordinary container step."""
    workspace_parent = repo.parent
    return {
        "PATH": os.environ["PATH"],
        "HOME": str(home if home is not None else workspace_parent / "bootstrap-home"),
        "GITHUB_WORKSPACE": str(repo),
        # Point both trusted-config overrides inside the sandbox so tests can
        # never touch a real user/system git config.
        "GIT_CONFIG_GLOBAL": str(
            global_config if global_config is not None else workspace_parent / "global.gitconfig"
        ),
        "GIT_CONFIG_SYSTEM": str(
            system_config if system_config is not None else workspace_parent / "system.gitconfig"
        ),
    }


def _clean_git_env(home: Path) -> dict[str, str]:
    return {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "GIT_CONFIG_GLOBAL": str(home / "gitconfig"),
        "GIT_CONFIG_SYSTEM": str(home / "system.gitconfig"),
    }


def _run_bootstrap(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )


def _git_result(
    repo: Path,
    env: dict[str, str],
    *args: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )


def _init_repo(repo: Path, credentials_home: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    commands = [
        ["git", "-C", str(repo), "init", "-q"],
        ["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
        ["git", "-C", str(repo), "config", "user.name", "AA test"],
        ["git", "-C", str(repo), "config", "commit.gpgsign", "false"],
        ["git", "-C", str(repo), "commit", "-q", "--allow-empty", "-m", "init"],
    ]
    env = _clean_git_env(credentials_home)
    for command in commands:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
            check=False,
        )
        if proc.returncode != 0:
            pytest.fail(f"test sandbox git setup failed: {proc.stderr}")


# --- workflow-level contracts ---------------------------------------------------


@pytest.mark.parametrize(("workflow_path", "jobs"), sorted(AFFECTED_JOBS.items()))
def test_affected_container_jobs_bootstrap_after_checkout(
    workflow_path: str, jobs: tuple[str, ...]
) -> None:
    workflow = _read(WORKFLOWS / workflow_path)
    for job in jobs:
        steps = _job_steps(workflow, job)
        checkout_indices = [i for i, step in enumerate(steps) if "actions/checkout@" in step]
        bootstrap_indices = [i for i, step in enumerate(steps) if BOOTSTRAP_RUN in step]
        assert len(checkout_indices) == 1, f"unexpected checkout count in {workflow_path}:{job}"
        assert bootstrap_indices, f"missing bootstrap invocation in {workflow_path}:{job}"
        assert bootstrap_indices[0] == checkout_indices[0] + 1, (
            f"{workflow_path}:{job} must invoke the shared bootstrap directly after checkout"
        )
        # Later re-invocations (e.g. a dedicated idempotency step) stay allowed
        # but must never precede the checkout.
        assert all(index > checkout_indices[0] for index in bootstrap_indices)
        pre_bootstrap = "\n".join(steps[: bootstrap_indices[0]])
        assert not _GIT_USAGE_RE.search(pre_bootstrap), (
            f"{workflow_path}:{job} runs git before the Git trust bootstrap"
        )


@pytest.mark.parametrize(("workflow_path", "jobs"), sorted(AFFECTED_JOBS.items()))
def test_affected_jobs_run_inside_dedicated_container(
    workflow_path: str, jobs: tuple[str, ...]
) -> None:
    workflow = _read(WORKFLOWS / workflow_path)
    for job in jobs:
        prefix = _job_prefix(workflow, job)
        assert "runs-on: ubuntu-latest" in prefix
        assert "container:" in prefix, f"{workflow_path}:{job} is expected to run in a container"


def test_script_avoids_all_forbidden_security_bypasses() -> None:
    text = _read(SCRIPT)
    for forbidden in (
        "safe.directory=*",
        "safe.directory=/*",
        "GIT_CONFIG_NOSYSTEM",
        "chown -R",
        "chmod -R",
    ):
        assert forbidden not in text
    # No wildcard value may ever be handed to `git config ... safe.directory`
    # and the script must reject workspaces containing wildcard characters.
    assert not re.search(r'safe\.directory[\s"=]+\*', text)
    assert "GITHUB_WORKSPACE" in text


def test_script_adds_only_the_exact_workspace_and_verifies_itself() -> None:
    text = _read(SCRIPT)
    assert 'run_git_config "$scope_flag" --add safe.directory "$workspace"' in text
    assert 'git -C "$workspace" rev-parse --show-toplevel' in text
    assert 'git -C "$workspace" rev-parse HEAD' in text
    assert 'git -C "$workspace" status --porcelain' in text
    assert "--show-origin" in text


@pytest.mark.parametrize("workflow_path", sorted(AFFECTED_JOBS))
def test_affected_workflows_do_not_weaken_git_security(workflow_path: str) -> None:
    text = _read(WORKFLOWS / workflow_path)
    for forbidden in ("safe.directory=*", "GIT_CONFIG_NOSYSTEM", "chown -R", "chmod -R"):
        assert forbidden not in text, f"{workflow_path} must never use {forbidden}"


def test_regression_workflow_is_a_real_read_only_pinned_container_check() -> None:
    text = _read(WORKFLOWS / "aa-container-git-trust-regression.yml")
    assert VERIFIED_IMAGE_DIGEST in text
    assert not re.search(r"ghcr\.io/kodmial/aa-runtime@sha256:0{64}", text)
    # Read-only evidence job: no book/LLM/model/Telegram secrets may exist.
    for forbidden in (
        "TELEGRAM_BOT_TOKEN",
        "AA_BOOK_AGE_IDENTITY",
        "OPENCODE_MODEL",
        "AA_BOT_PRIMARY_MODEL",
    ):
        assert forbidden not in text
    assert "pull_request:" in text
    steps = _job_steps(text, "checkout-bootstrap-verify")
    checkout_indices = [i for i, step in enumerate(steps) if "actions/checkout@" in step]
    bootstrap_indices = [i for i, step in enumerate(steps) if BOOTSTRAP_RUN in step]
    independent_git_indices = [
        i
        for i, step in enumerate(steps)
        if i >= bootstrap_indices[0] and "git rev-parse HEAD" in step
    ]
    assert checkout_indices and bootstrap_indices and independent_git_indices
    assert bootstrap_indices[0] == checkout_indices[0] + 1
    assert independent_git_indices[0] > bootstrap_indices[0]


# --- behavior of the shared script ----------------------------------------------


def test_bootstrap_adds_exact_entry_and_is_idempotent(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    repo = tmp_path / "repo"
    _init_repo(repo, home)
    global_config = tmp_path / "global.gitconfig"
    env = _sandbox_env(repo, home=home, global_config=global_config)

    first = _run_bootstrap(env)
    assert first.returncode == 0, "\n".join((first.stdout, first.stderr))
    get_first = _git_result(
        repo, env, "config", "--file", global_config.as_posix(), "--get-all", "safe.directory"
    )
    assert get_first.returncode == 0
    assert get_first.stdout.strip().splitlines() == [repo.as_posix()]

    second = _run_bootstrap(env)
    assert second.returncode == 0, "\n".join((second.stdout, second.stderr))
    get_second = _git_result(
        repo, env, "config", "--file", global_config.as_posix(), "--get-all", "safe.directory"
    )
    assert get_second.stdout.strip().splitlines() == [repo.as_posix()]

    # Ordinary git operations of later steps succeed against the trusted repo:
    # they read exactly the same effective trusted config as the bootstrap.
    post_env = dict(env)
    del post_env["GITHUB_WORKSPACE"]
    head = _git_result(repo, post_env, "rev-parse", "HEAD")
    assert head.returncode == 0, head.stderr
    assert re.fullmatch(r"[0-9a-f]{40}", head.stdout.strip())
    status = _git_result(repo, post_env, "status", "--porcelain")
    assert status.returncode == 0, status.stderr
    toplevel = _git_result(repo, post_env, "rev-parse", "--show-toplevel")
    assert toplevel.stdout.strip() == repo.as_posix()


def test_missing_workspace_fails_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    env = _sandbox_env(tmp_path / "unused-repo")
    del env["GITHUB_WORKSPACE"]
    proc = _run_bootstrap(env)
    assert proc.returncode != 0
    assert "GITHUB_WORKSPACE" in proc.stderr
    assert not (tmp_path / "global.gitconfig").exists()


def test_relative_workspace_fails_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    env = _sandbox_env(Path("relative-path"))
    del env["GITHUB_WORKSPACE"]  # replaced by a relative override below
    env["GITHUB_WORKSPACE"] = "relative-path"
    proc = _run_bootstrap(env)
    assert proc.returncode != 0
    assert "absolute path" in proc.stderr


def test_non_repository_root_fails_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "untrusted-marker.txt").write_text("not-a-repo", encoding="utf-8")
    env = _sandbox_env(plain)
    proc = _run_bootstrap(env)
    assert proc.returncode != 0
    assert "not a git repository root" in proc.stderr
    assert not (tmp_path / "global.gitconfig").exists()


def test_noncanonical_symlink_workspace_fails_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    repo = tmp_path / "repo"
    _init_repo(repo, home)
    link = tmp_path / "linked"
    link.symlink_to(repo)
    env = _sandbox_env(link)
    proc = _run_bootstrap(env)
    assert proc.returncode != 0
    assert "canonical" in proc.stderr


def test_wildcard_workspace_never_trusted(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    env = _sandbox_env(tmp_path / "star-*")
    proc = _run_bootstrap(env)
    assert proc.returncode != 0
    assert "wildcard" in proc.stderr


def test_unwritable_trusted_config_fails_clearly(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    repo = tmp_path / "repo"
    _init_repo(repo, home)
    env = {
        "PATH": os.environ["PATH"],
        "GITHUB_WORKSPACE": str(repo),
        "HOME": str(tmp_path / "nonexistent-bootstrap-home"),
        "GIT_CONFIG_SYSTEM": str(tmp_path / "nonexistent-parent" / "system.gitconfig"),
    }
    proc = _run_bootstrap(env)
    assert proc.returncode != 0
    assert "No writable trusted" in proc.stderr
    assert not (tmp_path / "global.gitconfig").exists()


# --- real differing-owner reproduction (the qualification defect) ----------------


def _chown_to_foreign_uid(path: Path) -> int | None:
    """Best-effort recursive chown of path to a uid differing from ours."""
    euid = os.geteuid()
    foreign = next(candidate for candidate in (65534, 1002, 1000, 1001) if candidate != euid)
    specification = f"{foreign}"
    if euid == 0:
        command = ["chown", "-R", specification, str(path)]
    else:
        if shutil.which("sudo") is None:
            return None
        command = ["sudo", "-n", "chown", "-R", specification, str(path)]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    return foreign if proc.returncode == 0 else None


def test_differing_owner_repository_is_trusted_after_bootstrap(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    repo = tmp_path / "repo"
    _init_repo(repo, home)
    assert _chown_to_foreign_uid(repo) is not None, (
        "cannot create a differing-owner repository in this environment"
    )
    pre = _git_result(repo, _clean_git_env(tmp_path / "pre-home"), "rev-parse", "HEAD")
    if pre.returncode != 0:
        # The genuine defect must be reproducible before trust is granted.
        assert "dubious ownership" in (pre.stdout + pre.stderr)
    env = _sandbox_env(repo)
    proc = _run_bootstrap(env)
    assert proc.returncode == 0, "\n".join((proc.stdout, proc.stderr))
    # Later ordinary steps run with the same effective HOME/config resolution.
    post_env = dict(env)
    del post_env["GITHUB_WORKSPACE"]
    post_head = _git_result(repo, post_env, "rev-parse", "HEAD")
    assert post_head.returncode == 0, post_head.stderr
    assert re.fullmatch(r"[0-9a-f]{40}", post_head.stdout.strip())
    post_status = _git_result(repo, post_env, "status", "--porcelain")
    assert post_status.returncode == 0, post_status.stderr
    toplevel = _git_result(repo, post_env, "rev-parse", "--show-toplevel")
    assert toplevel.stdout.strip() == repo.as_posix()
