"""Characterization tests for uav_nav_lab.recording.experiment_runner.

Verify the subprocess argv, the rmtree pre-step, and that
``extra_env`` is merged into the inherited environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from uav_nav_lab.recording import run_uav_nav_experiment


def test_run_uav_nav_experiment_wipes_results_dir_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    results = tmp_path / "results"
    results.mkdir()
    (results / "old_artifact.json").write_text("{}")

    monkeypatch.setattr(
        "uav_nav_lab.recording.experiment_runner.subprocess.run",
        lambda *a, **k: None,
    )

    run_uav_nav_experiment(tmp_path / "exp.yaml", results, repo_root=tmp_path)
    assert not results.exists()  # rmtree happened


def test_run_uav_nav_experiment_passes_yaml_path_into_python_dash_c(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    captured: dict[str, Any] = {}

    def fake_run(cmd: list[str], *, cwd: Any = None, check: bool = False, env: Any = None) -> None:
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env

    monkeypatch.setattr("uav_nav_lab.recording.experiment_runner.subprocess.run", fake_run)

    yaml = tmp_path / "my_exp.yaml"
    run_uav_nav_experiment(yaml, tmp_path / "nonexistent_results", repo_root=tmp_path)

    cmd = captured["cmd"]
    # python -c "<code>"
    assert cmd[1] == "-c"
    # The -c payload should reference the YAML path and the uav-nav CLI
    assert str(yaml) in cmd[2]
    assert "from uav_nav_lab.cli import main" in cmd[2]
    assert "'uav-nav','run'" in cmd[2]
    # cwd is the repo_root we asked for
    assert captured["cwd"] == tmp_path
    # No extra_env → env passthrough (subprocess inherits)
    assert captured["env"] is None


def test_run_uav_nav_experiment_merges_extra_env_with_os_environ(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("PRE_EXISTING", "hello")

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "uav_nav_lab.recording.experiment_runner.subprocess.run",
        lambda cmd, cwd=None, check=False, env=None: captured.update({"env": env}),
    )

    run_uav_nav_experiment(
        tmp_path / "exp.yaml",
        tmp_path / "results_dir",
        repo_root=tmp_path,
        extra_env={"UAV_NAV_NO_CAMERA": "1"},
    )

    env = captured["env"]
    assert env is not None
    assert env["PRE_EXISTING"] == "hello"          # inherited
    assert env["UAV_NAV_NO_CAMERA"] == "1"         # added
