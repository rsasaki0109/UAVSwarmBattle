"""Characterization tests for the ROS 2 bridge coord helpers.

The bridge's full publish-spin-read behaviour is already covered by
``test_ros2_bridge.py`` against a fake adapter. These tests pin the
extracted pure helpers (``_enu_to_ned`` / ``_ned_to_enu``) and the
thin bridge wrappers (``_from_ros_frame`` / ``_to_ros_frame``) so a
regression in the swap-and-flip mapping surfaces independently of the
bridge's larger plumbing.
"""

from __future__ import annotations

import numpy as np
import pytest

from uav_nav_lab.sim.ros2_bridge import _enu_to_ned, _ned_to_enu


def test_enu_to_ned_maps_x_y_z_to_y_x_minus_z():
    out = _enu_to_ned(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(out, [2.0, 1.0, -3.0])


def test_ned_to_enu_maps_y_x_minus_z_to_x_y_z():
    out = _ned_to_enu(np.array([2.0, 1.0, -3.0]))
    np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])


def test_ned_to_enu_inverts_enu_to_ned_round_trip():
    p = np.array([1.5, -2.5, 7.25])
    np.testing.assert_array_almost_equal(_ned_to_enu(_enu_to_ned(p)), p)


def test_enu_to_ned_pads_2d_input_with_zero_z():
    out = _enu_to_ned(np.array([4.0, 5.0]))
    np.testing.assert_array_equal(out, [5.0, 4.0, -0.0])


def test_enu_to_ned_truncates_4d_input_to_first_three_components():
    out = _enu_to_ned(np.array([1.0, 2.0, 3.0, 99.0]))
    np.testing.assert_array_equal(out, [2.0, 1.0, -3.0])


def _make_bridge(frame: str):
    from types import SimpleNamespace

    from uav_nav_lab.sim.ros2_bridge import Ros2Bridge

    sc = SimpleNamespace(start=np.array([0.0, 0.0]), goal=np.array([1.0, 1.0]),
                         ndim=2, occupancy=np.zeros((2, 2)))
    return Ros2Bridge(dt=0.05, scenario=sc, frame=frame, adapter=object())


def test_from_ros_frame_is_identity_in_enu_mode():
    b = _make_bridge("enu")
    v = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(b._from_ros_frame(v), v)


def test_to_ros_frame_is_identity_in_enu_mode():
    b = _make_bridge("enu")
    v = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(b._to_ros_frame(v), v)


def test_from_ros_frame_swaps_axes_in_ned_mode():
    # Adapter feeds NED-ordered (y, x, -z); bridge should hand the planner ENU.
    b = _make_bridge("ned")
    out = b._from_ros_frame(np.array([2.0, 1.0, -3.0]))
    np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])


def test_to_ros_frame_swaps_axes_in_ned_mode():
    # Bridge takes an ENU command and must hand the adapter NED.
    b = _make_bridge("ned")
    out = b._to_ros_frame(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(out, [2.0, 1.0, -3.0])


def test_bridge_rejects_unknown_frame():
    from types import SimpleNamespace
    from uav_nav_lab.sim.ros2_bridge import Ros2Bridge
    sc = SimpleNamespace(start=np.zeros(2), goal=np.ones(2), ndim=2,
                         occupancy=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="frame must be 'enu' or 'ned'"):
        Ros2Bridge(dt=0.05, scenario=sc, frame="foo", adapter=object())


def test_bridge_rejects_unknown_cmd_msg_type():
    from types import SimpleNamespace
    from uav_nav_lab.sim.ros2_bridge import Ros2Bridge
    sc = SimpleNamespace(start=np.zeros(2), goal=np.ones(2), ndim=2,
                         occupancy=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="cmd_msg_type must be 'twist' or 'airsim_vel_cmd'"):
        Ros2Bridge(dt=0.05, scenario=sc, cmd_msg_type="bogus", adapter=object())
