from __future__ import annotations

import numpy as np

from uav_nav_lab.sensor import SENSOR_REGISTRY
from uav_nav_lab.sensor.dropout import DropoutSensor


def _peers(n):
    return [{"position": [float(i), 0.0], "velocity": [0.0, 0.0], "radius": 0.4}
            for i in range(n)]


def test_dropout_registered() -> None:
    assert SENSOR_REGISTRY.get("dropout") is DropoutSensor


def test_dropout_zero_passes_everything_through() -> None:
    s = DropoutSensor.from_config({"dropout_prob": 0.0})
    s.reset(seed=0)
    peers = _peers(5)
    out = s.observe_dynamics(0.0, np.zeros(2), peers)
    assert len(out) == 5
    assert [d["position"] for d in out] == [p["position"] for p in peers]


def test_dropout_one_drops_everything() -> None:
    s = DropoutSensor.from_config({"dropout_prob": 1.0})
    s.reset(seed=0)
    assert s.observe_dynamics(0.0, np.zeros(2), _peers(5)) == []


def test_dropout_fraction_is_approximately_p_and_seed_reproducible() -> None:
    # Over many draws the kept fraction tracks (1 - p); same seed => same result.
    s = DropoutSensor.from_config({"dropout_prob": 0.5})
    s.reset(seed=42)
    kept = sum(len(s.observe_dynamics(0.0, np.zeros(2), _peers(10))) for _ in range(400))
    assert 0.45 < kept / (400 * 10) < 0.55

    a = DropoutSensor.from_config({"dropout_prob": 0.5}); a.reset(seed=7)
    b = DropoutSensor.from_config({"dropout_prob": 0.5}); b.reset(seed=7)
    pa = [len(a.observe_dynamics(0.0, np.zeros(2), _peers(8))) for _ in range(50)]
    pb = [len(b.observe_dynamics(0.0, np.zeros(2), _peers(8))) for _ in range(50)]
    assert pa == pb


def test_dropout_ego_pose_is_ground_truth() -> None:
    s = DropoutSensor.from_config({"dropout_prob": 1.0})
    s.reset(seed=0)
    pos = np.array([3.0, -2.0])
    assert np.allclose(s.observe(0.0, pos), pos)
