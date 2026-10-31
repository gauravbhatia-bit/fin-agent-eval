"""Cross-platform project tasks for Windows and other development systems."""

from __future__ import annotations

import argparse
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
VENV_PYTHON = (
    VENV_DIR / "Scripts" / "python.exe"
    if sys.platform == "win32"
    else VENV_DIR / "bin" / "python"
)


def run(command: list[str]) -> None:
    """Run a command from the project root and propagate its exit status."""
    subprocess.run(command, cwd=ROOT, check=True)


def setup() -> None:
    """Create the Python 3.12 environment and install the locked dependencies."""
    if sys.version_info[:2] != (3, 12):
        raise SystemExit(
            "Run setup with Python 3.12, for example: py -3.12 tasks.py setup"
        )
    if not VENV_DIR.exists():
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    run([str(VENV_PYTHON), "-m", "pip", "install", "-r", "requirements.txt"])
    run(
        [
            str(VENV_PYTHON),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "-e",
            ".",
        ]
    )
    run([str(VENV_PYTHON), "-m", "pre_commit", "install"])


def lint() -> None:
    """Run Ruff lint and formatting checks."""
    run([str(VENV_PYTHON), "-m", "ruff", "check", "."])
    run([str(VENV_PYTHON), "-m", "ruff", "format", "--check", "."])


def test() -> None:
    """Run the test suite with coverage for the application and evaluation code."""
    run(
        [
            str(VENV_PYTHON),
            "-m",
            "pytest",
            "-q",
            "--basetemp",
            str(ROOT / ".pytest-basetemp"),
            "--cov=app",
            "--cov=evals",
            "--cov-fail-under=75",
        ]
    )


def run_api() -> None:
    """Run the development API server."""
    run(
        [
            str(VENV_PYTHON),
            "-m",
            "uvicorn",
            "app.main:app",
            "--reload",
            "--port",
            "8000",
        ]
    )


def run_ui() -> None:
    """Run the Streamlit UI against BACKEND_URL (default: local API)."""
    run(
        [
            str(VENV_PYTHON),
            "-m",
            "streamlit",
            "run",
            "ui/app.py",
        ]
    )


def eval_smoke() -> None:
    """Run the small replay smoke suite; matching cassettes must be recorded."""
    run(
        [
            str(VENV_PYTHON),
            "-m",
            "evals.runner",
            "--subset",
            "smoke",
            "--mode",
            "replay",
            "--allow-draft",
        ]
    )


def eval_full() -> None:
    """Run the approved full golden suite against the configured live model."""
    run(
        [
            str(VENV_PYTHON),
            "-m",
            "evals.runner",
            "--subset",
            "full",
            "--mode",
            "live",
        ]
    )


def eval_record() -> None:
    """Record the smoke suite with live calls for later deterministic replay."""
    run(
        [
            str(VENV_PYTHON),
            "-m",
            "evals.runner",
            "--subset",
            "smoke",
            "--mode",
            "record",
            "--allow-draft",
        ]
    )


def gate() -> None:
    """Compare the latest evaluation report with the quality baseline."""
    run([str(VENV_PYTHON), "-m", "evals.gate"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task",
        choices=[
            "setup",
            "lint",
            "test",
            "run-api",
            "run-ui",
            "eval-smoke",
            "eval-full",
            "eval-record",
            "gate",
        ],
    )
    task = parser.parse_args().task
    tasks = {
        "setup": setup,
        "lint": lint,
        "test": test,
        "run-api": run_api,
        "run-ui": run_ui,
        "eval-smoke": eval_smoke,
        "eval-full": eval_full,
        "eval-record": eval_record,
        "gate": gate,
    }
    tasks[task]()


if __name__ == "__main__":
    main()
