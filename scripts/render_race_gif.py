"""Render a side-by-side top-down GIF of the drone-race oval circuit
with a bouncing dynamic obstacle, comparing 2 or 3 planners.

Loads `episode_NNN_drone_*.json` from each run directory, computes the
oval reference polyline, animates the 4 drones around it, and overlays
the dynamic obstacle (analytically recomputed from CLI params since it
is not logged per-step).

Usage (2-pane):
    python3 scripts/render_race_gif.py \\
        --runs results/race_oval4_mpc:MPC \\
               results/race_oval4_gpu_mppi:GPU\\ MPPI \\
        --out docs/images/compare_race_oval4.gif

Usage (3-pane):
    python3 scripts/render_race_gif.py \\
        --runs results/race_oval4_mpc:MPC \\
               results/race_oval4_gpu_mppi:GPU\\ MPPI \\
               results/race_oval4_gpu_mppi_smart_v4:Smart\\ MPPI\\ v4 \\
        --out docs/images/compare_race_oval4.gif
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers '3d' projection)
import numpy as np


DRONE_COLORS = ["#e8443b", "#3aa54a", "#3865bf", "#d49b1c"]
OBSTACLE_COLOR = "#cc1f1f"


def load_drones(run_dir: Path, ep: int, n_drones: int = 4) -> list[dict]:
    drones = []
    for i in range(n_drones):
        p = run_dir / f"episode_{ep:03d}_drone_{i:02d}.json"
        drones.append(json.loads(p.read_text()))
    return drones


def trajectory_arrays(
    drones: list[dict], T_pad: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (true_pos[D,T,3], ref_pos[D,T,3], collision_step[D]) arrays.
    `collision_step[i]` = step index where drone i first reports a
    collision flag, or T (never) if no flag was raised.

    When `T_pad` is given (e.g. the longest drone-episode across all
    panes), shorter trajectories are right-padded by holding their last
    logged position — so a planner that fails fast freezes in place
    instead of vanishing from the GIF mid-way."""
    D = len(drones)
    T_drones = min(len(d["steps"]) for d in drones)
    T = T_pad if T_pad is not None and T_pad > T_drones else T_drones
    true_pos = np.zeros((D, T, 3))
    ref_pos = np.zeros((D, T, 3))
    collision_step = np.full(D, T, dtype=int)
    for i, d in enumerate(drones):
        last_true = None
        last_ref = None
        for k in range(T):
            if k < len(d["steps"]):
                s = d["steps"][k]
                last_true = s["true_pos"]
                last_ref = s.get("reference_pos", s["true_pos"])
                if collision_step[i] == T and s.get("collision"):
                    collision_step[i] = k
            true_pos[i, k] = last_true
            ref_pos[i, k] = last_ref
        if collision_step[i] == T and d.get("outcome") == "collision":
            collision_step[i] = len(d["steps"]) - 1
    return true_pos, ref_pos, collision_step


def load_rollout_replans(drones: list[dict]) -> list[list[tuple[float, np.ndarray]]]:
    """For each drone, return a sorted list of (t, rollouts[K,H+1,3])
    tuples extracted from the replan log. Replans without a `rollouts`
    field (e.g. MPC) yield an empty list."""
    out: list[list[tuple[float, np.ndarray]]] = []
    for d in drones:
        seq: list[tuple[float, np.ndarray]] = []
        for r in d.get("replans", []):
            rl = r.get("rollouts")
            if rl is None:
                continue
            arr = np.asarray(rl, dtype=float)
            if arr.ndim != 3 or arr.shape[-1] != 3:
                continue
            seq.append((float(r["t"]), arr))
        seq.sort(key=lambda x: x[0])
        out.append(seq)
    return out


def current_rollouts(
    replans: list[tuple[float, np.ndarray]], t_s: float
) -> np.ndarray | None:
    """Return the most recent rollout cloud at time `t_s` (None if no
    replan with rollouts has happened yet)."""
    chosen: np.ndarray | None = None
    for t_r, arr in replans:
        if t_r > t_s:
            break
        chosen = arr
    return chosen


def obstacle_trajectory(
    start: np.ndarray,
    velocity: np.ndarray,
    dt: float,
    n_steps: int,
    world_size: np.ndarray,
) -> np.ndarray:
    """Recompute the dynamic obstacle trajectory analytically — the
    runner does not log it per-step. Mirrors `_DynamicObstacle3D.step`."""
    traj = np.zeros((n_steps, 3))
    pos = start.astype(float).copy()
    vel = velocity.astype(float).copy()
    for k in range(n_steps):
        traj[k] = pos
        pos = pos + vel * dt
        for i in range(3):
            upper = float(world_size[i])
            if pos[i] < 0:
                pos[i] = -pos[i]
                vel[i] = -vel[i]
            elif pos[i] > upper:
                pos[i] = 2 * upper - pos[i]
                vel[i] = -vel[i]
    return traj


def oval_polyline(
    center: np.ndarray, radius_x: float, radius_y: float, n_pts: int = 200
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0, 2 * np.pi, n_pts)
    x = center[0] + radius_x * np.cos(theta)
    y = center[1] + radius_y * np.sin(theta)
    z = np.full_like(theta, center[2])
    return x, y, z


def setup_axis(ax, world: np.ndarray, center: np.ndarray):
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # Hide z axis labels — pure top-down view
    ax.set_zticklabels([])
    ax.set_zlabel("")
    ax.set_xlim(0, world[0])
    ax.set_ylim(0, world[1])
    ax.set_zlim(0, world[2])
    # Near-pure top-down view: very high elevation, neutral azimuth.
    # elev=90 is degenerate in matplotlib 3D so we use 89.
    ax.view_init(elev=89, azim=-90)


def parse_run(s: str) -> tuple[Path, str]:
    """Parse `path:label` (label optional)."""
    if ":" in s:
        path_s, label = s.rsplit(":", 1)
    else:
        path_s, label = s, Path(s).name
    return Path(path_s), label


def episode_outcomes(drones: list[dict]) -> list[str]:
    return [d.get("outcome", "?") for d in drones]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="One or more `path:label` (label optional)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--trail", type=int, default=40)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--obstacle-start", type=float, nargs=3, default=[20.0, 8.0, 7.0])
    ap.add_argument("--obstacle-vel", type=float, nargs=3, default=[0.0, 6.0, 0.0])
    ap.add_argument("--obstacle-radius", type=float, default=1.2)
    ap.add_argument("--world", type=float, nargs=3, default=[40.0, 40.0, 14.0])
    ap.add_argument("--config", default=None,
                    help="Optional path to scenario YAML — when given, all "
                         "scenario.dynamic_obstacles entries are loaded and "
                         "rendered (overrides --obstacle-* args).")
    ap.add_argument("--title", default="Drone race (4 drones, oval circuit) + bouncing intruder",
                    help="Title prefix shown above the panes (time stamp appended).")
    args = ap.parse_args()

    runs = [parse_run(r) for r in args.runs]
    n_panes = len(runs)
    if n_panes < 2 or n_panes > 4:
        raise SystemExit("--runs must list 2..4 entries")

    all_drones: list[list[dict]] = []
    rollouts_per_pane: list[list[list[tuple[float, np.ndarray]]]] = []
    for run_dir, _ in runs:
        drones = load_drones(run_dir, args.ep)
        all_drones.append(drones)
        rollouts_per_pane.append(load_rollout_replans(drones))

    # Pad shorter panes to the longest pane's timeline by holding final
    # positions — so planners that fail fast freeze in place instead of
    # truncating every other pane.
    T_pad = max(
        min(len(d["steps"]) for d in pane_drones) for pane_drones in all_drones
    )
    true_arr: list[np.ndarray] = []
    ref_arr: list[np.ndarray] = []
    coll_step_arr: list[np.ndarray] = []
    for pane_drones in all_drones:
        true_p, ref_p, coll_step = trajectory_arrays(pane_drones, T_pad=T_pad)
        true_arr.append(true_p)
        ref_arr.append(ref_p)
        coll_step_arr.append(coll_step)

    # Recover oval geometry from the first run's reference trajectory
    ref0 = ref_arr[0][0]
    center = np.array(
        [float(ref0[:, 0].mean()), float(ref0[:, 1].mean()), float(ref0[:, 2].mean())]
    )
    rx = float((ref0[:, 0].max() - ref0[:, 0].min()) / 2.0)
    ry = float((ref0[:, 1].max() - ref0[:, 1].min()) / 2.0)
    world = np.asarray(args.world, dtype=float)
    ox, oy, oz = oval_polyline(center, rx, ry)

    # Dynamic obstacle trajectories (analytical). Sourced either from a
    # config YAML (all `scenario.dynamic_obstacles` entries) or from the
    # single-obstacle `--obstacle-*` args.
    T_max = min(t.shape[1] for t in true_arr)
    obstacles: list[dict] = []
    if args.config:
        import yaml as _yaml
        cfg = _yaml.safe_load(Path(args.config).read_text())
        for d in cfg.get("scenario", {}).get("dynamic_obstacles", []) or []:
            obstacles.append({
                "start": np.asarray(d["start"], dtype=float),
                "vel": np.asarray(d["velocity"], dtype=float),
                "radius": float(d.get("radius", 0.5)),
            })
    if not obstacles:
        obstacles.append({
            "start": np.asarray(args.obstacle_start, dtype=float),
            "vel": np.asarray(args.obstacle_vel, dtype=float),
            "radius": float(args.obstacle_radius),
        })
    obs_trajs = [
        obstacle_trajectory(o["start"], o["vel"], args.dt, T_max, world)
        for o in obstacles
    ]

    # How many rollout polylines to pre-allocate per (pane, drone). Read
    # the first non-empty rollout cloud to size the pool.
    max_k = 0
    for pane_repl in rollouts_per_pane:
        for drone_repl in pane_repl:
            for _, arr in drone_repl:
                if arr.shape[0] > max_k:
                    max_k = arr.shape[0]

    # Auto-shrink per-pane width so the 4-pane variant fits under the
    # 2000px image API limit while keeping the 2-pane / 3-pane variants
    # at their original 6-inch-per-pane proportions.
    pane_w = 6.0 if n_panes <= 3 else 4.5
    fig = plt.figure(figsize=(pane_w * n_panes + 1, 5.5))
    axes: list = []
    trail_lines: list[list] = []
    drone_pts: list[list] = []
    rollout_lines: list[list[list]] = []  # [pane][drone][k]
    obs_trail_artists: list = []
    obs_pt_artists: list = []
    ref_pt_artists: list = []
    ref_line_artists: list = []
    for pane in range(n_panes):
        ax = fig.add_subplot(1, n_panes, pane + 1, projection="3d")
        setup_axis(ax, world, center)
        ax.plot(ox, oy, oz, color="#666666", linewidth=1.0, alpha=0.7,
                linestyle="--")
        axes.append(ax)
        pane_trails: list = []
        pane_pts: list = []
        pane_rollouts: list[list] = []
        pane_ref_pts: list = []
        pane_ref_lines: list = []
        for i in range(4):
            c = DRONE_COLORS[i]
            # Rollout cloud (drawn under the drone marker for layering).
            drone_rollouts: list = []
            if max_k > 0:
                for _ in range(max_k):
                    ln_r, = ax.plot([], [], [], color=c, linewidth=0.5,
                                    alpha=0.18)
                    drone_rollouts.append(ln_r)
            pane_rollouts.append(drone_rollouts)
            ln, = ax.plot([], [], [], color=c, linewidth=1.8, alpha=0.9)
            pt, = ax.plot([], [], [], "o", color=c, markersize=9,
                          markeredgecolor="white", markeredgewidth=0.6)
            # Reference "ghost" marker — where the drone *would* be if it
            # were tracking the oval reference exactly. Faded same-colour
            # ring with no fill; visible deviation = visible avoidance.
            ref_pt, = ax.plot([], [], [], "o", color=c, markersize=7,
                              markerfacecolor="none", markeredgewidth=1.2,
                              alpha=0.55)
            # Line connecting reference to actual position — makes the
            # magnitude/direction of deviation visually unambiguous.
            ref_ln, = ax.plot([], [], [], color=c, linewidth=1.0,
                              alpha=0.35, linestyle=":")
            pane_trails.append(ln)
            pane_pts.append(pt)
            pane_ref_pts.append(ref_pt)
            pane_ref_lines.append(ref_ln)
        trail_lines.append(pane_trails)
        drone_pts.append(pane_pts)
        rollout_lines.append(pane_rollouts)
        ref_pt_artists.append(pane_ref_pts)
        ref_line_artists.append(pane_ref_lines)
        pane_obs_trails: list = []
        pane_obs_pts: list = []
        # Render each obstacle as a square sized to match its physical
        # radius (in world units, projected to point-space). The world is
        # 40 m wide and each pane is `pane_w` inches at 100 dpi, so 1 m
        # ≈ 2.5 * pane_w points. Sphere diameter in pts ≈ 5 * pane_w * r.
        px_per_m = 2.5 * pane_w
        for ob in obstacles:
            ms = max(6.0, 2.0 * ob["radius"] * px_per_m)
            ot, = ax.plot([], [], [], color=OBSTACLE_COLOR, linewidth=0.8, alpha=0.35)
            op, = ax.plot([], [], [], "s", color=OBSTACLE_COLOR,
                          markersize=ms, markeredgecolor="black", markeredgewidth=1.0)
            pane_obs_trails.append(ot)
            pane_obs_pts.append(op)
        obs_trail_artists.append(pane_obs_trails)
        obs_pt_artists.append(pane_obs_pts)

    title_text = fig.suptitle("", fontsize=13)
    frames = list(range(0, T_max, args.stride))

    flash_window = 12  # frames after first-collision flag to flash white
    empty_xyz = np.zeros((0,), dtype=float)

    def update(k: int):
        k0 = max(0, k - args.trail)
        t_s = k * args.dt
        artists: list = []
        for pane, (_, label) in enumerate(runs):
            tp = true_arr[pane]
            coll_step = coll_step_arr[pane]
            rp = ref_arr[pane]
            for i in range(4):
                trail_lines[pane][i].set_data(tp[i, k0:k+1, 0], tp[i, k0:k+1, 1])
                trail_lines[pane][i].set_3d_properties(tp[i, k0:k+1, 2])
                drone_pts[pane][i].set_data(tp[i, k:k+1, 0], tp[i, k:k+1, 1])
                drone_pts[pane][i].set_3d_properties(tp[i, k:k+1, 2])
                # Reference ghost marker + deviation line.
                ref_pt_artists[pane][i].set_data(
                    rp[i, k:k+1, 0], rp[i, k:k+1, 1])
                ref_pt_artists[pane][i].set_3d_properties(rp[i, k:k+1, 2])
                ref_line_artists[pane][i].set_data(
                    [rp[i, k, 0], tp[i, k, 0]],
                    [rp[i, k, 1], tp[i, k, 1]])
                ref_line_artists[pane][i].set_3d_properties(
                    [rp[i, k, 2], tp[i, k, 2]])
                # Collision flash: white-out the marker for `flash_window`
                # frames after the first collision flag, then revert to a
                # dimmed grey ghost.
                base_c = DRONE_COLORS[i]
                dk = k - int(coll_step[i])
                if dk < 0:
                    drone_pts[pane][i].set_color(base_c)
                    drone_pts[pane][i].set_alpha(0.9)
                elif dk < flash_window:
                    drone_pts[pane][i].set_color("#ffffff")
                    drone_pts[pane][i].set_alpha(1.0)
                else:
                    drone_pts[pane][i].set_color("#888888")
                    drone_pts[pane][i].set_alpha(0.55)
                # Rollout cloud: refresh from the most recent replan with
                # rollouts that happened at or before t_s. Hide unused
                # polylines for this drone.
                cloud = current_rollouts(rollouts_per_pane[pane][i], t_s)
                for kk, ln_r in enumerate(rollout_lines[pane][i]):
                    if cloud is not None and kk < cloud.shape[0]:
                        ln_r.set_data(cloud[kk, :, 0], cloud[kk, :, 1])
                        ln_r.set_3d_properties(cloud[kk, :, 2])
                    else:
                        ln_r.set_data(empty_xyz, empty_xyz)
                        ln_r.set_3d_properties(empty_xyz)
                artists.extend(rollout_lines[pane][i])
            for o_idx, otraj in enumerate(obs_trajs):
                obs_trail_artists[pane][o_idx].set_data(
                    otraj[k0:k+1, 0], otraj[k0:k+1, 1])
                obs_trail_artists[pane][o_idx].set_3d_properties(otraj[k0:k+1, 2])
                obs_pt_artists[pane][o_idx].set_data(otraj[k:k+1, 0], otraj[k:k+1, 1])
                obs_pt_artists[pane][o_idx].set_3d_properties(otraj[k:k+1, 2])
            err = np.linalg.norm(tp[:, k, :] - ref_arr[pane][:, k, :], axis=1).mean()
            # Live collision counter: # drones whose first-collision step
            # has already passed at this frame.
            n_coll_live = int((coll_step < k).sum())
            outcomes = episode_outcomes(all_drones[pane])
            n_coll_final = sum(1 for o in outcomes if o == "collision")
            axes[pane].set_title(
                f"{label}   err {err:.2f} m   coll {n_coll_live}/4 (final {n_coll_final}/4)",
                fontsize=12,
            )
            artists.extend(trail_lines[pane])
            artists.extend(drone_pts[pane])
            artists.extend(ref_pt_artists[pane])
            artists.extend(ref_line_artists[pane])
            artists.extend(obs_trail_artists[pane])
            artists.extend(obs_pt_artists[pane])
        title_text.set_text(f"{args.title}   t = {t_s:.2f} s")
        artists.append(title_text)
        return artists

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 // args.fps, blit=False)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=args.fps)
    anim.save(out_path, writer=writer)
    plt.close(fig)
    print(f"wrote {out_path}  ({len(frames)} frames @ {args.fps} fps, {n_panes} panes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
