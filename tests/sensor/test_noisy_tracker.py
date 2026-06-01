from __future__ import annotations

import numpy as np

from uav_nav_lab.sensor.noisy_tracker import NoisyTrackerSensor


def _obs(y: float) -> list[dict]:
    return [{"position": [10.0, y], "velocity": [0.0, 5.0], "radius": 2.0}]


def test_zero_noise_zero_delay_is_passthrough() -> None:
    # delay 0 → buffer_len 1 → leftmost == latest; no noise → identity.
    s = NoisyTrackerSensor.from_config({"delay": 0.0, "dt": 0.05})
    s.reset(seed=1)
    rep = s.observe_dynamics(0.0, np.zeros(2), _obs(10.0))
    assert rep[0]["position"] == [10.0, 10.0]
    assert rep[0]["velocity"] == [0.0, 5.0]
    assert rep[0]["radius"] == 2.0


def test_delay_reports_stale_position() -> None:
    # delay 0.15 / dt 0.05 → 3-step buffer. After feeding y=10,11,12,13 the
    # reported position should lag to the oldest still in the window (y=11),
    # not the latest (y=13).
    s = NoisyTrackerSensor.from_config({"delay": 0.15, "dt": 0.05})
    s.reset(seed=0)
    ys = [10.0, 11.0, 12.0, 13.0]
    rep = None
    for y in ys:
        rep = s.observe_dynamics(0.0, np.zeros(2), _obs(y))
    assert rep is not None
    # buffer maxlen=3 holds [11,12,13]; leftmost = 11.
    assert rep[0]["position"][1] == 11.0


def test_position_noise_is_seeded_reproducible() -> None:
    cfg = {"delay": 0.0, "dt": 0.05, "position_noise_std": 1.0}
    a = NoisyTrackerSensor.from_config(cfg)
    a.reset(seed=42)
    b = NoisyTrackerSensor.from_config(cfg)
    b.reset(seed=42)
    ra = a.observe_dynamics(0.0, np.zeros(2), _obs(10.0))
    rb = b.observe_dynamics(0.0, np.zeros(2), _obs(10.0))
    assert ra[0]["position"] == rb[0]["position"]
    # and actually perturbed away from ground truth
    assert ra[0]["position"][1] != 10.0


def test_velocity_noise_perturbs_only_when_enabled() -> None:
    s_off = NoisyTrackerSensor.from_config({"position_noise_std": 1.0})
    s_off.reset(seed=3)
    rep = s_off.observe_dynamics(0.0, np.zeros(2), _obs(10.0))
    assert rep[0]["velocity"] == [0.0, 5.0]  # untouched

    s_on = NoisyTrackerSensor.from_config({"velocity_noise_std": 2.0})
    s_on.reset(seed=3)
    rep = s_on.observe_dynamics(0.0, np.zeros(2), _obs(10.0))
    assert rep[0]["velocity"] != [0.0, 5.0]


def test_reset_clears_per_obstacle_buffers() -> None:
    # After a reset the delay buffer must not leak the previous episode's
    # stale samples — the first post-reset report should be the fresh truth.
    s = NoisyTrackerSensor.from_config({"delay": 0.15, "dt": 0.05})
    s.reset(seed=0)
    for y in (10.0, 11.0, 12.0):
        s.observe_dynamics(0.0, np.zeros(2), _obs(y))
    s.reset(seed=0)
    rep = s.observe_dynamics(0.0, np.zeros(2), _obs(99.0))
    assert rep[0]["position"][1] == 99.0


def test_handles_obstacle_count_growth() -> None:
    # A scenario that adds an obstacle mid-run must not IndexError; a new
    # buffer is created on demand.
    s = NoisyTrackerSensor.from_config({"delay": 0.1, "dt": 0.05})
    s.reset(seed=0)
    s.observe_dynamics(0.0, np.zeros(2), _obs(10.0))
    two = _obs(10.0) + [{"position": [5.0, 5.0], "velocity": [1.0, 0.0], "radius": 1.0}]
    rep = s.observe_dynamics(0.05, np.zeros(2), two)
    assert len(rep) == 2
    assert rep[1]["position"] == [5.0, 5.0]
