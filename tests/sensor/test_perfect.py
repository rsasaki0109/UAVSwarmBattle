from __future__ import annotations

import numpy as np

from uav_nav_lab.sensor.perfect import PerfectSensor


def test_perfect_sensor_can_bias_dynamic_velocity_for_controls() -> None:
    sensor = PerfectSensor.from_config(
        {"type": "perfect", "dynamic_velocity_scale": -1.0}
    )

    observed = sensor.observe_dynamics(
        0.0,
        np.array([0.0, 0.0, 0.0]),
        [
            {
                "position": [1.0, 2.0, 3.0],
                "velocity": [0.0, 1.5, 0.0],
                "radius": 1.0,
            }
        ],
    )

    assert observed[0]["position"] == [1.0, 2.0, 3.0]
    assert observed[0]["velocity"] == [-0.0, -1.5, -0.0]
    assert observed[0]["radius"] == 1.0
