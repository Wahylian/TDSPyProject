"""Install dependencies: try the GPU (CUDA) torch build, fall back to CPU.

pip's requirements files are a static list with no "try this wheel, else that
one" logic, so a plain `pip install -r requirements.txt` just fails outright if
the pinned cu126 wheels aren't available for this machine/platform. This script
provides the fallback: try requirements.txt (GPU) first, and on failure retry
with requirements-cpu.txt.

    python install.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent


def _pip_install(requirements_file: str) -> bool:
    """Run `pip install -r <requirements_file>`; return True on success."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(PROJECT_DIR / requirements_file)]
    )
    return result.returncode == 0


def main() -> None:
    print("Installing dependencies (GPU torch build)...")
    if _pip_install("requirements.txt"):
        print("Done: installed with the GPU (CUDA) torch build.")
        return

    print("\nGPU install failed; falling back to the CPU torch build...")
    if _pip_install("requirements-cpu.txt"):
        print("Done: installed with the CPU torch build.")
        return

    raise SystemExit("Both the GPU and CPU installs failed; see the pip output above.")


if __name__ == "__main__":
    main()
