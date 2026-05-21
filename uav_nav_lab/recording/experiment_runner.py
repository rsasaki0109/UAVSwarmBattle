"""Subprocess wrapper for ``uav-nav run`` inside recording scripts.

All five recorders shell out to ``uav-nav`` via Python so the
subprocess inherits a clean import state — the parent script can
``import airsim`` and set up the camera without bleeding airsim
state into the experiment runner. This wrapper bundles the
``shutil.rmtree`` of the prior run dir + the ``subprocess.run`` call.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _build_cli_command(yaml_path: Path) -> list[str]:
    """Build the ``python -c`` argv that invokes the uav-nav CLI.

    Kept as a single string passed to ``-c`` (not a wrapper script) so
    the recorder doesn't need to know where the ``uav-nav`` console
    script is installed — `python -m uav_nav_lab.cli` would also work,
    but the CLI lives under :func:`uav_nav_lab.cli.main` and the
    explicit invocation is what the original 5 scripts settled on.
    """
    return [
        sys.executable,
        "-c",
        (
            "import sys; sys.argv=['uav-nav','run',str(r'%s')];"
            "from uav_nav_lab.cli import main; main()" % yaml_path
        ),
    ]


def run_uav_nav_experiment(
    yaml_path: Path,
    results_dir: Path,
    *,
    repo_root: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> None:
    """Run ``uav-nav run <yaml_path>`` in a subprocess.

    Wipes ``results_dir`` first so each recording starts from a clean
    slate. ``extra_env`` is merged into the current environment for
    flags like ``UAV_NAV_NO_CAMERA=1`` (used by the top-down recorder
    so the framework's in-loop camera capture is skipped while the
    script captures frames externally).

    ``repo_root`` defaults to the current working directory; the
    scripts pass an explicit ``REPO_ROOT`` so they work regardless of
    where they were invoked from.
    """
    if results_dir.exists():
        shutil.rmtree(results_dir)
    cwd = repo_root if repo_root is not None else Path.cwd()
    cmd = _build_cli_command(yaml_path)
    if extra_env is None:
        subprocess.run(cmd, cwd=cwd, check=True)
    else:
        env = {**os.environ, **extra_env}
        subprocess.run(cmd, cwd=cwd, check=True, env=env)
