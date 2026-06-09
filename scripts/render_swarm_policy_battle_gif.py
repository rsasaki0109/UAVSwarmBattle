#!/usr/bin/env python3
"""3-way battle montage from swarm_policy_battle results.

Picks a seed where the champion succeeds and challengers fail (from
phase.json if available, else defaults), then renders a side-by-side GIF
with a tactical HUD: radar rings, collision flashes, arena scoreboard.

  python scripts/render_swarm_policy_battle_gif.py
  python scripts/render_swarm_policy_battle_gif.py --seed 6003 --arms mpc_gt swarm_transformer
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from uav_nav_lab.config import ExperimentConfig  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE = REPO_ROOT / "results/swarm_policy_battle/phase.json"
OUT = REPO_ROOT / "docs/images/swarm_policy_battle_obstacle.gif"

import render_swarm_transformer_obstacle_gif as rs  # noqa: E402
import swarm_policy_battle_phase as battle  # noqa: E402

# Tactical palette — darker, higher-contrast than the lab default.
BG = "#060a08"
PANEL = "#0c1410"
RADAR = "#2d6a3e"
GRID = "#1a2e22"
HUB = "#4a7c59"
MISSILE_COLORS = ("#ff4500", "#ff2244", "#ffaa00")
EXHAUST_COLORS = ("#ffcc00", "#ff8844", "#ffe066")
PREDICT_COLORS = ("#ff6b6b", "#ff4466", "#ff9955")
MISSILE = MISSILE_COLORS[0]
EXHAUST = EXHAUST_COLORS[0]
PREDICT = PREDICT_COLORS[0]
THREAT_HALO = "#ff3b30"
WIN_C = "#3dff7a"
LOSS_C = "#ff453a"
WARN_C = "#ffb020"
FLASH_C = "#ffe566"
EVADE_C = "#58d8ff"
LOCK_C = "#ff2244"
MONO = "monospace"

COLL_RADIUS = 0.8
MISSILE_RADIUS = 1.5
HUB_X, HUB_Y = 25.0, 25.0
ARENA = 50.0
SIM_DT = 0.05
LOCK_RANGE = 11.0
EVADE_RANGE = 6.5
IMPACT_RANGE = COLL_RADIUS + MISSILE_RADIUS

ARM_LABELS = {
    "orca": "ORCA (stock)",
    "orca_conv": "ORCA + convention",
    "hrvo": "HRVO",
    "mgr": "MGR",
    "mpc_gt": "MPC + GT (teacher)",
    "swarm_transformer": "swarm_transformer",
    "navrl": "NavRL (upstream ckpt)",
}


def _pick_seed(
    phase_path: Path,
    scenario: str,
    ref: str,
    challengers: list[str],
) -> int | None:
    if not phase_path.is_file():
        return None
    data = json.loads(phase_path.read_text())
    cell = data.get("cells", {}).get(scenario, {})
    by_arm = cell.get("outcomes_by_seed", {})
    if ref not in by_arm:
        return None
    seeds = sorted(int(s) for s in by_arm[ref].keys())
    for s in seeds:
        if by_arm[ref].get(str(s)) != "success":
            continue
        if all(by_arm.get(c, {}).get(str(s)) != "success" for c in challengers):
            return s
    return None


def _arena_scores(phase_path: Path, scenario: str) -> dict[str, tuple[int, int]]:
    if not phase_path.is_file():
        return {}
    data = json.loads(phase_path.read_text())
    arms = data.get("cells", {}).get(scenario, {}).get("arms", {})
    return {k: (int(v.get("joint_ok", 0)), int(v.get("n", 20))) for k, v in arms.items()}


def _cfg(scenario: str) -> ExperimentConfig:
    raw = battle._cfg(scenario, "orca", seed=6000, n_eps=1)
    return ExperimentConfig.from_dict(raw)


def _pad(traj: list[np.ndarray], n: int) -> list[np.ndarray]:
    out = list(traj)
    while len(out) < n:
        out.append(out[-1])
    return out


def _status_tag(joint: str) -> tuple[str, str]:
    if joint == "success":
        return "VICTORY", WIN_C
    if joint == "collision":
        return "DOWN", LOSS_C
    return "JAMMED", WARN_C


def _finite_velocity(hist: list[np.ndarray], f: int, default: np.ndarray) -> np.ndarray:
    default = np.asarray(default, float)
    if len(hist) < 2:
        return default.copy()
    i0 = max(0, f - 1)
    i1 = min(f, len(hist) - 1)
    if i0 == i1:
        i0 = max(0, i1 - 1)
    dv = np.asarray(hist[i1], float) - np.asarray(hist[i0], float)
    dt = max(i1 - i0, 1) * SIM_DT
    v = dv / dt
    if v.ndim == 1:
        if float(np.linalg.norm(v)) < 0.05:
            return default.copy()
        return v
    norms = np.linalg.norm(v, axis=1)
    fill = default if default.ndim == 1 else default[0]
    low = norms < 0.05
    if np.any(low):
        v = v.copy()
        v[low] = fill
    return v


def _flight_start(obs_hist: list[np.ndarray], starts: np.ndarray) -> int:
    """First frame where any missile has clearly left its pad."""
    for f, pts in enumerate(obs_hist):
        if pts.size == 0:
            continue
        for m in range(min(pts.shape[0], len(starts))):
            if float(np.linalg.norm(pts[m] - starts[m])) > 2.0:
                return max(0, f - 3)
    return 0


def _hub_drama_center(obs_hist: list[np.ndarray]) -> int:
    """Frame where missiles are tightest around the hub — peak crossfire."""
    best_f, best_score = 0, float("inf")
    hub = np.array([HUB_X, HUB_Y])
    for f, pts in enumerate(obs_hist):
        if pts.size == 0:
            continue
        d = float(np.min(np.linalg.norm(pts - hub, axis=1)))
        if d < best_score:
            best_score, best_f = d, f
    return best_f


def _predict_missile_path(pos: np.ndarray, vel: np.ndarray, *, steps: int = 10) -> np.ndarray:
    p = np.asarray(pos, float).copy()
    v = np.asarray(vel, float).copy()
    pts = [p.copy()]
    for _ in range(steps):
        p = p + v * SIM_DT
        if p[0] <= 0.0 or p[0] >= ARENA:
            v[0] *= -1.0
            p[0] = float(np.clip(p[0], 0.0, ARENA))
        if p[1] <= 0.0 or p[1] >= ARENA:
            v[1] *= -1.0
            p[1] = float(np.clip(p[1], 0.0, ARENA))
        pts.append(p.copy())
    return np.stack(pts)


def _nearest_drone(pos: np.ndarray, drones: np.ndarray) -> tuple[int, float]:
    d = np.linalg.norm(drones - pos, axis=1)
    i = int(np.argmin(d))
    return i, float(d[i])


def _nearest_threat(
    obs_pts: np.ndarray, drones: np.ndarray,
) -> tuple[int, int, float]:
    best_i, best_m, best_d = 0, 0, float("inf")
    for m in range(obs_pts.shape[0]):
        i, d = _nearest_drone(obs_pts[m], drones)
        if d < best_d:
            best_i, best_m, best_d = i, m, d
    return best_i, best_m, best_d


def _missile_defaults() -> list[np.ndarray]:
    return [np.asarray(d["velocity"], float) for d in battle.DYN_OBS]


def _hub_crossings(obs_hist: list[np.ndarray], f: int) -> int:
    crosses = 0
    upto = min(f + 1, len(obs_hist))
    for i in range(1, upto):
        p0 = np.asarray(obs_hist[i - 1], float)
        p1 = np.asarray(obs_hist[i], float)
        if p0.ndim == 1:
            p0, p1 = p0.reshape(1, 2), p1.reshape(1, 2)
        for m in range(p0.shape[0]):
            y0, y1 = float(p0[m, 1]), float(p1[m, 1])
            x1 = float(p1[m, 0])
            if (y0 - HUB_Y) * (y1 - HUB_Y) < 0.0 and abs(x1 - HUB_X) < 4.0:
                crosses += 1
    return crosses


def _collision_hotspots(pos: np.ndarray, obs_pts: np.ndarray | None) -> np.ndarray:
    hot: list[np.ndarray] = []
    for i in range(pos.shape[0]):
        for j in range(i + 1, pos.shape[0]):
            if float(np.linalg.norm(pos[i] - pos[j])) < COLL_RADIUS:
                hot.extend([pos[i], pos[j]])
        if obs_pts is not None:
            for m in range(obs_pts.shape[0]):
                o = obs_pts[m]
                if float(np.linalg.norm(pos[i] - o)) < IMPACT_RANGE:
                    hot.append(pos[i])
                    hot.append(o)
    return np.array(hot) if hot else np.empty((0, 2))


def _lock_bracket(ax, center: np.ndarray, half: float = 1.1) -> list:
    x, y = float(center[0]), float(center[1])
    h = half
    specs = [
        ([x - h, x - h + 0.55], [y - h, y - h]),
        ([x - h, x - h], [y - h, y - h + 0.55]),
        ([x + h - 0.55, x + h], [y - h, y - h]),
        ([x + h, x + h], [y - h, y - h + 0.55]),
        ([x - h, x - h + 0.55], [y + h, y + h]),
        ([x - h, x - h], [y + h - 0.55, y + h]),
        ([x + h - 0.55, x + h], [y + h, y + h]),
        ([x + h, x + h], [y + h - 0.55, y + h]),
    ]
    return [ax.plot(xs, ys, color=LOCK_C, lw=1.6, alpha=0.95, zorder=8)[0] for xs, ys in specs]


def _set_quiver(qv, pos: np.ndarray, vel: np.ndarray, *, floor: float = 0.4) -> None:
    v = np.asarray(vel, float)
    pos = np.asarray(pos, float)
    if v.ndim == 1:
        if float(np.linalg.norm(v)) < floor:
            v = np.array([floor, 0.0])
        qv.set_offsets(pos.reshape(1, 2))
        qv.set_UVC(v[0], v[1])
        return
    vn = np.linalg.norm(v, axis=1)
    uv = v.copy()
    low = vn < floor
    if np.any(low):
        uv[low] = np.array([floor, 0.0])
    qv.set_offsets(pos)
    qv.set_UVC(uv[:, 0], uv[:, 1])


def _draw_tactical_hud(ax, *, obstacle: bool) -> list:
    """Static radar overlay; returns animated artists (sweep line)."""
    for x in range(0, 51, 10):
        ax.axvline(x, color=GRID, lw=0.35, alpha=0.55, zorder=0)
    for y in range(0, 51, 10):
        ax.axhline(y, color=GRID, lw=0.35, alpha=0.55, zorder=0)
    ax.axhline(HUB_Y, color=RADAR, lw=0.5, alpha=0.35, ls=":", zorder=1)
    ax.axvline(HUB_X, color=RADAR, lw=0.5, alpha=0.35, ls=":", zorder=1)
    for r in (6.0, 12.0, 18.0):
        from matplotlib.patches import Circle
        ax.add_patch(Circle((HUB_X, HUB_Y), r, fill=False, ec=RADAR, lw=0.7,
                            ls=(0, (4, 6)), alpha=0.45, zorder=1))
    if obstacle:
        from matplotlib.patches import Circle
        ax.add_patch(Circle((HUB_X, HUB_Y), 3.0, fill=False, ec=HUB, lw=1.4,
                            ls="--", alpha=0.75, zorder=2))
        ax.text(HUB_X, HUB_Y - 4.2, "INTERCEPT ZONE", color=HUB, fontsize=6,
                ha="center", va="top", family=MONO, alpha=0.85, zorder=2)
        launch_sites = (
            (HUB_X, 2.0, "S"),
            (2.0, HUB_Y, "W"),
            (48.0, HUB_Y, "E"),
        )
        for idx, (lx, ly, tag) in enumerate(launch_sites):
            y_off = -2.4 if tag == "S" else (2.4 if tag == "W" else -2.4)
            x_off = 0.0 if tag == "S" else (-2.4 if tag == "W" else 2.4)
            ax.text(lx + x_off, ly + y_off, f"PAD {tag}",
                    color=MISSILE_COLORS[idx], fontsize=4.5,
                    ha="center", family=MONO, alpha=0.55, zorder=2)
    # corner brackets
    b = 2.5
    corners = [
        ([b, b + 4], [b, b]), ([b, b], [b, b + 4]),
        ([50 - b - 4, 50 - b], [b, b]), ([50 - b, 50 - b], [b, b + 4]),
        ([b, b + 4], [50 - b, 50 - b]), ([b, b], [50 - b - 4, 50 - b]),
        ([50 - b - 4, 50 - b], [50 - b, 50 - b]), ([50 - b, 50 - b], [50 - b - 4, 50 - b]),
    ]
    for xs, ys in corners:
        ax.plot(xs, ys, color=RADAR, lw=1.2, alpha=0.7, zorder=2)
    sweep, = ax.plot([HUB_X, HUB_X + 16], [HUB_Y, HUB_Y], color=RADAR, lw=1.0,
                     alpha=0.55, zorder=2)
    return [sweep]


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Circle

    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="obstacle", choices=("peers", "obstacle"))
    ap.add_argument("--ref", default="swarm_transformer")
    ap.add_argument("--arms", nargs="+", default=["navrl", "mpc_gt", "swarm_transformer"])
    ap.add_argument("--seed", type=int, default=0, help="0 = auto-pick from phase.json")
    ap.add_argument("--phase", default=str(PHASE))
    ap.add_argument("--fps", type=int, default=18)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=160)
    ap.add_argument("--start-frame", type=int, default=-1,
                    help="animation start (-1 = auto: missiles in flight)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    challengers = [a for a in args.arms if a != args.ref]
    phase_path = Path(args.phase)
    seed = args.seed
    if seed == 0:
        seed = _pick_seed(phase_path, args.scenario, args.ref, challengers) or 6003

    scores = _arena_scores(phase_path, args.scenario)
    cfg = _cfg(args.scenario)
    arm_by_label = {ARM_LABELS.get(a, a): a for a in args.arms}

    runs = {}
    for arm in args.arms:
        runs[ARM_LABELS.get(arm, arm)] = rs._rollout(
            cfg, seed, planner=battle._planner(arm, args.scenario),
        )

    nf = min(max(len(r.drones) for r in runs.values()), 400)
    ref_run = runs[ARM_LABELS.get(args.ref, args.ref)]
    starts = np.stack([np.asarray(d["start"][:2], float) for d in battle.DYN_OBS])
    if args.start_frame >= 0:
        f0 = args.start_frame
    else:
        f0 = _flight_start(ref_run.obstacles, starts)
        drama = _hub_drama_center(ref_run.obstacles)
        f0 = max(f0, max(0, drama - args.max_frames // 2))
    f1 = min(nf, f0 + args.max_frames)
    frames = list(range(f0, f1, max(1, args.stride)))
    n = next(iter(runs.values())).goals.shape[0]
    colors = plt.cm.plasma(np.linspace(0.12, 0.88, n))

    n_arms = len(args.arms)
    fig, axes = plt.subplots(1, n_arms, figsize=(4.6 * n_arms, 5.4))
    if n_arms == 1:
        axes = [axes]
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.06, wspace=0.10)
    art: dict = {}
    sweeps: list = []

    for ax, (title, run) in zip(axes, runs.items()):
        ax.set_facecolor(PANEL)
        ax.set_aspect("equal")
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 50)
        ax.set_xticks([])
        ax.set_yticks([])

        tag, tag_c = _status_tag(run.joint)
        arm_key = arm_by_label.get(title, "")
        ok_n, total_n = scores.get(arm_key, (0, 20))
        border_c = tag_c
        for sp in ax.spines.values():
            sp.set_color(border_c)
            sp.set_linewidth(2.2)

        sweeps.extend(_draw_tactical_hud(ax, obstacle=args.scenario == "obstacle"))
        ax.scatter(run.goals[:, 0], run.goals[:, 1], s=90, marker="*", c=colors,
                   edgecolors="#e6edf3", linewidths=0.3, alpha=0.55, zorder=3)

        drones = _pad(run.drones, nf)
        obs = _pad(run.obstacles, nf) if run.obstacles else []
        n_missiles = obs[0].shape[0] if obs else 0
        m_defaults = _missile_defaults()
        trails = [ax.plot([], [], "-", color=colors[i], lw=1.2, alpha=0.45, zorder=4)[0]
                  for i in range(n)]
        drone_q = ax.quiver(
            drones[0][:, 0], drones[0][:, 1], np.ones(n), np.zeros(n),
            color=colors, scale=30, width=0.007, headwidth=3.2, headlength=4.0,
            headaxislength=3.5, zorder=6,
        )
        missiles: list[dict] = []
        lock_lines: list = []
        alert_txt = ax.text(25, 47.5, "", color=LOCK_C, fontsize=7, ha="center",
                            family=MONO, fontweight="bold", zorder=9)
        evade_txt = ax.text(25, 3.8, "", color=EVADE_C, fontsize=6, ha="center",
                            family=MONO, fontweight="bold", zorder=9)
        if n_missiles:
            o0 = obs[0]
            for mi in range(n_missiles):
                mc = MISSILE_COLORS[mi % len(MISSILE_COLORS)]
                ec = EXHAUST_COLORS[mi % len(EXHAUST_COLORS)]
                pc = PREDICT_COLORS[mi % len(PREDICT_COLORS)]
                dv = m_defaults[mi] if mi < len(m_defaults) else np.array([0.0, 4.5])
                intercept_ring = Circle(tuple(o0[mi]), 2.8, fill=False, ec=THREAT_HALO,
                                        lw=0.9, ls=(0, (3, 3)), alpha=0.65, zorder=4)
                ax.add_patch(intercept_ring)
                kill_ring = Circle(tuple(o0[mi]), IMPACT_RANGE, fill=False, ec=LOCK_C,
                                   lw=0.8, alpha=0.35, zorder=4)
                ax.add_patch(kill_ring)
                missile_trail, = ax.plot([], [], "-", color=ec, lw=3.8, alpha=0.72, zorder=5)
                smoke_trail, = ax.plot([], [], "-", color=ec, lw=6.0, alpha=0.18, zorder=4)
                predict_line, = ax.plot([], [], ":", color=pc, lw=0.9, alpha=0.35, zorder=3)
                missile_q = ax.quiver(
                    o0[mi, 0], o0[mi, 1], dv[0], dv[1], color=mc, scale=14.0, width=0.012,
                    headwidth=3.8, headlength=4.8, headaxislength=4.0, zorder=8,
                )
                exhaust_q = ax.quiver(
                    o0[mi, 0], o0[mi, 1], -dv[0], -dv[1], color=ec, scale=32, width=0.028,
                    alpha=0.75, zorder=6,
                )
                body = ax.scatter(
                    [o0[mi, 0]], [o0[mi, 1]], s=220, marker="^", c=mc,
                    edgecolors="white", linewidths=0.6, zorder=9,
                )
                glow = Circle(tuple(o0[mi]), MISSILE_RADIUS + 0.3, fc=mc, ec="none", alpha=0.25, zorder=7)
                ax.add_patch(glow)
                label = ax.text(o0[mi, 0], o0[mi, 1] + 2.4, f"M{mi + 1}",
                                color=mc, fontsize=5, ha="center", family=MONO,
                                fontweight="bold", zorder=10)
                missiles.append(dict(
                    missile_q=missile_q, exhaust_q=exhaust_q, missile_trail=missile_trail,
                    smoke_trail=smoke_trail, predict_line=predict_line,
                    intercept_ring=intercept_ring, kill_ring=kill_ring,
                    label=label, body=body, glow=glow, default_vel=dv,
                    launch=np.array([o0[mi, 0], o0[mi, 1]], float),
                ))

        flash = ax.scatter([], [], s=520, facecolors="none", edgecolors=FLASH_C,
                           linewidths=3.0, zorder=10)
        blast = ax.scatter([], [], s=900, c=MISSILE, marker="*", alpha=0.0, zorder=10)

        ax.set_title(
            f"{title}\n▌ {tag}  {ok_n}/{total_n}",
            color=tag_c, fontsize=9, fontweight="bold", pad=6, family=MONO,
        )
        art[title] = dict(
            ax=ax, trails=trails, drone_q=drone_q, missiles=missiles, lock_lines=lock_lines,
            alert_txt=alert_txt, evade_txt=evade_txt, flash=flash, blast=blast,
            drones=drones, obs=obs, prev_lock_dist=None,
        )

    # VS markers between panes
    for i in range(n_arms - 1):
        x = (i + 1) / n_arms
        fig.text(x, 0.52, "VS", color=LOSS_C, fontsize=14, fontweight="bold",
                 ha="center", va="center", family=MONO,
                 bbox=dict(boxstyle="round,pad=0.15", fc=BG, ec=LOSS_C, lw=1.2, alpha=0.95))

    fig.suptitle(
        f"▶ UAV SWARM POLICY BATTLE · MISSILE DEFENSE · SEED {seed}",
        color="#e6edf3", fontsize=12, fontweight="bold", y=0.95, family=MONO,
    )
    fig.text(0.5, 0.90,
             f"triple crossfire ({len(battle.DYN_OBS)} missiles) · evade or collide · n=20 paired seeds",
             color=RADAR, fontsize=7, ha="center", family=MONO)

    def update(k):
        f = frames[k]
        angle = 2.0 * math.pi * (k / max(len(frames) - 1, 1))
        sweep_len = 17.0
        for sw in sweeps:
            sw.set_data([HUB_X, HUB_X + sweep_len * math.cos(angle)],
                        [HUB_Y, HUB_Y + sweep_len * math.sin(angle)])
        out = list(sweeps)
        for title in runs:
            a = art[title]
            pos = a["drones"][f]
            dvel = _finite_velocity(a["drones"], f, np.tile([1.0, 0.0], (pos.shape[0], 1)))
            _set_quiver(a["drone_q"], pos, dvel, floor=0.8)
            for i, tr in enumerate(a["trails"]):
                lo = max(0, f - 28)
                hist = np.array([a["drones"][j][i] for j in range(lo, f + 1)])
                tr.set_data(hist[:, 0], hist[:, 1])

            obs_pts = a["obs"][f] if f < len(a["obs"]) and a["obs"][f].size else None
            inbound = 0
            for mi, ms in enumerate(a["missiles"]):
                if obs_pts is None or mi >= obs_pts.shape[0]:
                    continue
                pt = obs_pts[mi]
                m_hist = [a["obs"][j][mi] for j in range(f + 1) if j < len(a["obs"])]
                obs_vel = _finite_velocity(m_hist, len(m_hist) - 1, ms["default_vel"])
                _set_quiver(ms["missile_q"], pt, obs_vel, floor=1.0)
                _set_quiver(ms["exhaust_q"], pt, -obs_vel, floor=1.0)
                ms["intercept_ring"].center = tuple(pt)
                ms["kill_ring"].center = tuple(pt)
                tail_lo = max(0, f - 32)
                tail = np.array(m_hist[tail_lo:], float)
                if len(tail) > 1:
                    ms["missile_trail"].set_data(tail[:, 0], tail[:, 1])
                    ms["smoke_trail"].set_data(tail[:, 0], tail[:, 1])
                pred = _predict_missile_path(pt, obs_vel, steps=8)
                ms["predict_line"].set_data(pred[:, 0], pred[:, 1])
                ms["body"].set_offsets([pt])
                ms["glow"].center = tuple(pt)
                spd = float(np.linalg.norm(obs_vel))
                ms["glow"].set_alpha(min(0.55, 0.2 + 0.04 * spd))
                ms["label"].set_position((pt[0], pt[1] + 2.4))
                dist_launch = float(np.linalg.norm(pt - ms["launch"]))
                if dist_launch < 1.2:
                    ms["label"].set_text(f"▶ FIRE M{mi + 1}")
                else:
                    ms["label"].set_text(f"M{mi + 1}")
                if float(np.min(np.linalg.norm(pos - pt, axis=1))) < LOCK_RANGE:
                    inbound += 1

            salvo = _hub_crossings(a["obs"], f) if a["obs"] else 0
            if obs_pts is not None and obs_pts.size:
                tgt_i, tgt_m, dist = _nearest_threat(obs_pts, pos)
                if a["lock_lines"]:
                    for ln in a["lock_lines"]:
                        ln.set_data([], [])
                if dist < LOCK_RANGE:
                    if not a["lock_lines"]:
                        a["lock_lines"].extend(_lock_bracket(a["ax"], pos[tgt_i]))
                    else:
                        x, y = float(pos[tgt_i, 0]), float(pos[tgt_i, 1])
                        h = 1.1
                        coords = [
                            ([x - h, x - h + 0.55], [y - h, y - h]),
                            ([x - h, x - h], [y - h, y - h + 0.55]),
                            ([x + h - 0.55, x + h], [y - h, y - h]),
                            ([x + h, x + h], [y - h, y - h + 0.55]),
                            ([x - h, x - h + 0.55], [y + h, y + h]),
                            ([x - h, x - h], [y + h - 0.55, y + h]),
                            ([x + h - 0.55, x + h], [y + h, y + h]),
                            ([x + h, x + h], [y + h - 0.55, y + h]),
                        ]
                        for ln, (xs, ys) in zip(a["lock_lines"], coords):
                            ln.set_data(xs, ys)
                    closing = (
                        a["prev_lock_dist"] is not None and dist < a["prev_lock_dist"] - 0.02
                    )
                    a["prev_lock_dist"] = dist
                    multi = f" x{inbound}" if inbound > 1 else ""
                    if dist < IMPACT_RANGE:
                        a["alert_txt"].set_text(f"◆ IMPACT M{tgt_m + 1}{multi} ◆")
                        a["alert_txt"].set_color(LOSS_C)
                    elif dist < EVADE_RANGE:
                        salvo_tag = f" · SALVO {salvo}" if salvo > 1 else ""
                        a["alert_txt"].set_text(f"INBOUND M{tgt_m + 1}{multi} — EVADE{salvo_tag}")
                        a["alert_txt"].set_color(LOCK_C)
                        a["evade_txt"].set_text("BREAK LOCK ↑" if closing else "DODGE ←→")
                    else:
                        salvo_tag = f" · WAVE {salvo}" if salvo > 1 else ""
                        a["alert_txt"].set_text(f"TRACK M{tgt_m + 1}{multi}{salvo_tag}")
                        a["alert_txt"].set_color(WARN_C)
                        a["evade_txt"].set_text("")
                else:
                    a["prev_lock_dist"] = None
                    a["alert_txt"].set_text(
                        f"{inbound} INBOUND" if inbound else "",
                    )
                    a["alert_txt"].set_color(WARN_C)
                    a["evade_txt"].set_text("")

            hot = _collision_hotspots(pos, obs_pts)
            a["flash"].set_offsets(hot)
            if len(hot):
                a["blast"].set_offsets(hot)
                a["blast"].set_alpha(0.85)
            else:
                a["blast"].set_offsets(np.empty((0, 2)))
                a["blast"].set_alpha(0.0)

            out += a["trails"] + [a["drone_q"], a["flash"], a["blast"], a["alert_txt"], a["evade_txt"]]
            for ms in a["missiles"]:
                out += [
                    ms["smoke_trail"], ms["missile_trail"], ms["predict_line"],
                    ms["glow"], ms["body"], ms["exhaust_q"], ms["missile_q"],
                    ms["intercept_ring"], ms["kill_ring"], ms["label"],
                ]
            out += a["lock_lines"]
        return out

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / args.fps, blit=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=PillowWriter(fps=args.fps),
              savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    print(f"[gif] {out}  ({out.stat().st_size // 1024} KB, seed {seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
