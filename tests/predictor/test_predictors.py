"""Predictor registry + per-predictor unit tests."""

from __future__ import annotations

import json  # noqa: F401
import subprocess  # noqa: F401
import sys  # noqa: F401
from pathlib import Path  # noqa: F401

import numpy as np  # noqa: F401
import pytest  # noqa: F401

from uav_nav_lab.cli import build_parser, main  # noqa: F401
from uav_nav_lab.config import ExperimentConfig  # noqa: F401
from uav_nav_lab.eval import evaluate_run  # noqa: F401
from uav_nav_lab.planner import PLANNER_REGISTRY  # noqa: F401
from uav_nav_lab.runner import expand_sweep, run_experiment  # noqa: F401

from tests._helpers import EXAMPLES, _basic_cfg, _require_mplot3d  # noqa: F401


def test_predictor_registry_has_defaults() -> None:
    from uav_nav_lab.predictor import PREDICTOR_REGISTRY

    names = PREDICTOR_REGISTRY.names()
    assert "constant_velocity" in names
    assert "noisy_velocity" in names


def test_constant_velocity_predictor_extrapolates_linearly() -> None:
    import numpy as np

    from uav_nav_lab.predictor import build_predictor

    p = build_predictor({"type": "constant_velocity"})
    obs = [{"position": [0.0, 0.0], "velocity": [1.0, 2.0], "radius": 0.5}]
    traj = p.predict(obs, np.array([1.0, 2.0, 3.0]))
    # one obstacle, three time points, 2D
    assert traj.shape == (1, 3, 2)
    assert np.allclose(traj[0, 0], [1.0, 2.0])
    assert np.allclose(traj[0, 2], [3.0, 6.0])


def test_kalman_predictor_basic_roundtrip() -> None:
    """Kalman predictor must (a) register, (b) produce the right shape,
    (c) eventually agree with the truth on a clean constant-velocity
    target after a few measurement updates."""
    import numpy as np

    from uav_nav_lab.predictor import build_predictor

    p = build_predictor({"type": "kalman_velocity", "dt": 0.1,
                         "process_noise_std": 0.1, "measurement_noise_std": 0.05})
    # Simulate a target moving at v=(2, 0) starting from (0, 0) for ~10 dt
    truth_v = np.array([2.0, 0.0])
    pos = np.zeros(2)
    horizon_dts = np.array([0.1, 0.5, 1.0])
    for _ in range(10):
        obs = [{"position": list(pos), "velocity": list(truth_v), "radius": 0.5}]
        traj = p.predict(obs, horizon_dts)
        assert traj.shape == (1, 3, 2)
        pos = pos + truth_v * 0.1
    # After 10 updates the KF velocity estimate should be close to truth
    track_v = p._tracks[0]["x"][2:]
    assert np.allclose(track_v, truth_v, atol=0.2)


def test_kalman_track_associates_across_calls() -> None:
    """A drifting target observed across multiple calls should remain a
    single track (not be re-spawned every call as a brand-new one)."""
    import numpy as np

    from uav_nav_lab.predictor import build_predictor

    p = build_predictor({"type": "kalman_velocity", "dt": 0.2,
                         "association_threshold": 5.0})
    horizon_dts = np.array([0.2])
    pos = np.array([10.0, 10.0])
    for _ in range(5):
        obs = [{"position": list(pos), "velocity": [1.0, 0.0], "radius": 0.5}]
        p.predict(obs, horizon_dts)
        pos[0] += 0.2  # drift x by dt·v = 0.2 per call
    assert len(p._tracks) == 1, "track was duplicated on each call"


def test_kalman_delayed_sensor_recovers_current_pose() -> None:
    """Kalman-delayed sensor should converge to the true current pose
    on a constant-velocity target after the buffer fills + a few KF updates."""
    import numpy as np

    from uav_nav_lab.sensor import SENSOR_REGISTRY

    cls = SENSOR_REGISTRY.get("kalman_delayed")
    sensor = cls.from_config({
        "delay": 0.2, "dt": 0.05,
        "process_noise_std": 0.5, "measurement_noise_std": 0.05,
    })
    sensor.reset(seed=42)
    pos = np.zeros(2)
    truth_v = np.array([2.0, 0.0])
    last_obs = None
    last_truth = None
    for k in range(40):
        last_truth = pos.copy()
        last_obs = sensor.observe(k * 0.05, pos)
        pos = pos + truth_v * 0.05
    # by step 40 the KF should be tracking close to the current truth
    assert np.allclose(last_obs, last_truth, atol=0.3)


def test_delayed_sensor_velocity_window_smooths_noisy_position() -> None:
    """A larger velocity_window should reduce the variance of the
    extrapolated estimate when position observations are noisy."""
    import numpy as np

    from uav_nav_lab.sensor import SENSOR_REGISTRY

    cls = SENSOR_REGISTRY.get("delayed")
    rng = np.random.default_rng(0)
    truth_v = np.array([2.0, 0.0])

    def run(window: int) -> float:
        sensor = cls.from_config({
            "delay": 0.2, "dt": 0.05, "extrapolate": True,
            "position_noise_std": 0.0, "velocity_window": window,
        })
        sensor.reset(seed=42)
        pos = np.zeros(2)
        outs = []
        for k in range(40):
            # noisy true position (sim of imperfect localization input)
            noisy_pos = pos + rng.normal(0.0, 0.05, size=2)
            outs.append(sensor.observe(k * 0.05, noisy_pos))
            pos = pos + truth_v * 0.05
        # measure variance of the estimate's deviation from truth
        outs = np.asarray(outs[10:])  # let buffer fill
        true_traj = np.stack([truth_v * (k * 0.05) for k in range(10, 40)])
        return float(np.std(outs - true_traj))

    err_w1 = run(window=1)
    err_w5 = run(window=5)
    # window=5 should produce a noticeably smaller error stdev than window=1
    assert err_w5 < err_w1, f"window=5 ({err_w5:.3f}) did not improve over window=1 ({err_w1:.3f})"


def test_delayed_sensor_extrapolate_recovers_current_pose() -> None:
    """With `extrapolate=True`, a stale measurement should be projected
    forward by `delay`, recovering close to the true current pose for a
    constant-velocity target."""
    import numpy as np

    from uav_nav_lab.sensor import SENSOR_REGISTRY

    cls = SENSOR_REGISTRY.get("delayed")
    sensor = cls.from_config({"delay": 0.1, "dt": 0.05, "extrapolate": True})
    sensor.reset()
    # constant-velocity true motion: pos = [t, 0]; v=(1,0)
    pos = np.zeros(2)
    last_obs = None
    for k in range(10):
        last_obs = sensor.observe(k * 0.05, pos)
        pos = pos + np.array([1.0, 0.0]) * 0.05
    # final true position is [10*0.05, 0] = [0.5, 0]
    # extrapolated obs should be close to current truth (within numerical noise)
    assert np.allclose(last_obs, [0.5, 0.0], atol=0.1)


def test_kalman_delay_compensation_extrapolates_forward() -> None:
    """With delay_compensation set, the output should sit ahead of the
    raw rollout by delay_compensation × velocity."""
    import numpy as np

    from uav_nav_lab.predictor import build_predictor

    base = build_predictor({"type": "kalman_velocity", "dt": 0.1,
                            "delay_compensation": 0.0})
    leaded = build_predictor({"type": "kalman_velocity", "dt": 0.1,
                              "delay_compensation": 0.5})
    obs = [{"position": [0.0, 0.0], "velocity": [3.0, 0.0], "radius": 0.5}]
    dts = np.array([0.1])
    a = base.predict(obs, dts)
    b = leaded.predict(obs, dts)
    # b should be 0.5 * 3.0 = 1.5 m further along x than a (with first-call
    # bootstrap velocity from observation)
    assert np.isclose(b[0, 0, 0] - a[0, 0, 0], 1.5, atol=0.05)


def test_noisy_predictor_seed_is_deterministic() -> None:
    import numpy as np

    from uav_nav_lab.predictor import build_predictor

    cfg = {"type": "noisy_velocity", "velocity_noise_std": 1.0}
    obs = [{"position": [0.0, 0.0], "velocity": [1.0, 0.0], "radius": 0.5}]
    dts = np.array([1.0, 2.0])
    p1 = build_predictor(cfg)
    p1.reset(seed=123)
    a = p1.predict(obs, dts)
    p2 = build_predictor(cfg)
    p2.reset(seed=123)
    b = p2.predict(obs, dts)
    assert np.allclose(a, b)
    # but a fresh seed should produce a different draw
    p3 = build_predictor(cfg)
    p3.reset(seed=456)
    c = p3.predict(obs, dts)
    assert not np.allclose(a, c)
