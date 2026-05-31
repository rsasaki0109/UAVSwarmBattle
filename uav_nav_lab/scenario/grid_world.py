"""2D grid-world scenario.

Coordinates are in meters; the underlying occupancy grid is integer-cell.
Each obstacle is a 1x1 cell. `world_resolution` lets you scale meters-per-cell
if you want a denser grid (default 1.0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .base import SCENARIO_REGISTRY, Scenario


@dataclass
class _ObstacleSpec:
    type: str = "random"
    count: int = 0
    seed: int = 0
    cells: list[tuple[int, int]] | None = None  # explicit list overrides random


@dataclass
class _DynamicObstacle:
    """Dynamic obstacle with a selectable motion policy.

    Policies
    --------
    ``linear``    : constant velocity, optionally reflecting at world bounds
                    (the original behaviour; ignores drone positions).
    ``pursue``    : steers toward the nearest drone at fixed ``speed``
                    (lead pursuit — always points at the target's *current*
                    position).
    ``intercept`` : proportional-navigation-style lead — aims at where the
                    target *will be*, using a finite-difference estimate of
                    the target's velocity. The "smart" pursuer.

    For pursue/intercept, ``set_targets`` must be called each step before
    ``step`` (the simulator and the animation replay both do this). The
    optional ``turn_rate`` (rad/s) caps how fast the heading can swing, which
    gives the chase visible inertia instead of snapping instantly.
    """

    pos0: np.ndarray
    velocity: np.ndarray
    reflect: bool = True
    radius: float = 0.5
    policy: str = "linear"
    speed: float | None = None  # pursue/intercept cruise speed; default |velocity|
    turn_rate: float | None = None  # max heading change rad/s (None = instant)
    pos: np.ndarray = None  # type: ignore[assignment]
    vel: np.ndarray = None  # type: ignore[assignment]
    _prev_target: np.ndarray | None = None  # for intercept lead estimation
    _prev_target_idx: int = -1  # which target the prev sample refers to

    def reset(self) -> None:
        self.pos = self.pos0.copy()
        self.vel = self.velocity.copy()
        self._prev_target = None
        self._prev_target_idx = -1

    def _cruise_speed(self) -> float:
        if self.speed is not None:
            return float(self.speed)
        s = float(np.linalg.norm(self.velocity))
        return s if s > 1e-9 else 1.0

    def _steer(self, dt: float, targets: list[np.ndarray] | None) -> None:
        """Update self.vel to chase the nearest target (pursue/intercept)."""
        if not targets:
            return  # no perception this step → coast on current velocity
        tgts = [np.asarray(t, dtype=float) for t in targets]
        nearest_idx = min(
            range(len(tgts)),
            key=lambda i: float(np.sum((tgts[i] - self.pos) ** 2)),
        )
        nearest = tgts[nearest_idx]
        aim = nearest
        # Only take the finite-difference lead when this step's nearest target
        # is the SAME one as last step's. If the nearest switched (another
        # target overtook, or a target dropped out), differencing two distinct
        # targets' positions would fabricate a huge bogus velocity, so we fall
        # back to pure pursuit for one step until the estimate restabilises.
        if (
            self.policy == "intercept"
            and self._prev_target is not None
            and self._prev_target_idx == nearest_idx
            and dt > 0
        ):
            # finite-difference target velocity, then lead by closing time.
            tvel = (nearest - self._prev_target) / dt
            speed = self._cruise_speed()
            dist = float(np.linalg.norm(nearest - self.pos))
            lead_t = dist / speed if speed > 1e-9 else 0.0
            lead_t = min(lead_t, 5.0)  # cap so an erratic target can't fling aim
            aim = nearest + tvel * lead_t
        self._prev_target = nearest.copy()
        self._prev_target_idx = nearest_idx
        direction = aim - self.pos
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            return
        desired = direction / norm * self._cruise_speed()
        if self.turn_rate is None:
            self.vel = desired
            return
        # rotate current heading toward desired by at most turn_rate*dt
        cur = self.vel
        cur_norm = float(np.linalg.norm(cur))
        if cur_norm < 1e-9:
            self.vel = desired
            return
        cu = cur / cur_norm
        du = desired / float(np.linalg.norm(desired))
        cos_a = float(np.clip(np.dot(cu, du), -1.0, 1.0))
        angle = float(np.arccos(cos_a))
        max_step = self.turn_rate * dt
        if angle <= max_step:
            self.vel = desired
            return
        t = max_step / angle
        blended = (1 - t) * cu + t * du
        bn = float(np.linalg.norm(blended))
        self.vel = blended / bn * self._cruise_speed() if bn > 1e-9 else desired

    def step(
        self,
        dt: float,
        world_size: tuple[float, ...],
        targets: list[np.ndarray] | None = None,
    ) -> None:
        if self.policy in ("pursue", "intercept"):
            self._steer(dt, targets)
        self.pos = self.pos + self.vel * dt
        if not self.reflect:
            return
        for i in range(len(self.pos)):
            upper = world_size[i]
            if self.pos[i] < 0:
                self.pos[i] = -self.pos[i]
                self.vel[i] = -self.vel[i]
            elif self.pos[i] > upper:
                self.pos[i] = 2 * upper - self.pos[i]
                self.vel[i] = -self.vel[i]


def _dynamic_from_specs(specs: Any) -> list[_DynamicObstacle]:
    """Build dynamic obstacles from a config list (shared by grid scenarios)."""
    out: list[_DynamicObstacle] = []
    for d in specs or []:
        out.append(
            _DynamicObstacle(
                pos0=np.asarray(d["start"], dtype=float),
                velocity=np.asarray(d.get("velocity", [0.0, 0.0]), dtype=float),
                reflect=bool(d.get("reflect", True)),
                radius=float(d.get("radius", 0.5)),
                policy=str(d.get("policy", "linear")),
                speed=(float(d["speed"]) if d.get("speed") is not None else None),
                turn_rate=(
                    float(d["turn_rate"]) if d.get("turn_rate") is not None else None
                ),
            )
        )
    return out


@SCENARIO_REGISTRY.register("grid_world")
class GridWorldScenario(Scenario):
    def __init__(
        self,
        size: tuple[int, int],
        start: tuple[float, float],
        goal: tuple[float, float],
        obstacles: _ObstacleSpec,
        resolution: float = 1.0,
        dynamic_obstacles: list[_DynamicObstacle] | None = None,
    ) -> None:
        self.size = size
        self.resolution = float(resolution)
        self._start = np.asarray(start, dtype=float)
        self._goal = np.asarray(goal, dtype=float)
        self._obs_spec = obstacles
        self._rng = np.random.default_rng(obstacles.seed)
        self._static_occ = np.zeros((size[0], size[1]), dtype=bool)
        self.occupancy = self._static_occ  # alias until advance() rebuilds
        self._dynamic: list[_DynamicObstacle] = list(dynamic_obstacles or [])
        self._targets: list[np.ndarray] = []
        self._populate()
        for d in self._dynamic:
            d.reset()
        self._refresh_occupancy()

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "GridWorldScenario":
        size = tuple(cfg.get("size", (50, 50)))
        if len(size) != 2:
            raise ValueError("scenario.size must be 2D")
        obs_cfg = dict(cfg.get("obstacles", {}))
        obstacles = _ObstacleSpec(
            type=str(obs_cfg.get("type", "random")),
            count=int(obs_cfg.get("count", 0)),
            seed=int(obs_cfg.get("seed", 0)),
            cells=obs_cfg.get("cells"),
        )
        dynamic = _dynamic_from_specs(cfg.get("dynamic_obstacles", []))
        return cls(
            size=(int(size[0]), int(size[1])),
            start=tuple(cfg.get("start", (1.0, 1.0))),
            goal=tuple(cfg.get("goal", (size[0] - 2, size[1] - 2))),
            obstacles=obstacles,
            resolution=float(cfg.get("resolution", 1.0)),
            dynamic_obstacles=dynamic,
        )

    def reseed(self, seed: int) -> None:
        # Mix run-level seed with scenario-level obstacle seed so multiple
        # episodes inside one run get different layouts deterministically.
        self._rng = np.random.default_rng(seed ^ self._obs_spec.seed)
        self._static_occ[:] = False
        self._targets = []
        self._populate()
        for d in self._dynamic:
            d.reset()
        self._refresh_occupancy()

    def set_targets(self, positions: list[np.ndarray]) -> None:
        """Record current drone positions for pursuing obstacles to chase."""
        self._targets = [np.asarray(p, dtype=float) for p in positions]

    def advance(self, dt: float) -> None:
        """Move dynamic obstacles forward by `dt` and refresh occupancy."""
        if not self._dynamic:
            return
        world = (self.size[0] * self.resolution, self.size[1] * self.resolution)
        for d in self._dynamic:
            d.step(dt, world, targets=self._targets)
        self._refresh_occupancy()

    def _refresh_occupancy(self) -> None:
        if not self._dynamic:
            self.occupancy = self._static_occ
            return
        grid = self._static_occ.copy()
        for d in self._dynamic:
            ix = int(d.pos[0] / self.resolution)
            iy = int(d.pos[1] / self.resolution)
            cells = max(1, int(np.ceil(d.radius / self.resolution)))
            for dx in range(-cells + 1, cells):
                for dy in range(-cells + 1, cells):
                    px, py = ix + dx, iy + dy
                    if 0 <= px < self.size[0] and 0 <= py < self.size[1]:
                        grid[px, py] = True
        self.occupancy = grid

    def _populate(self) -> None:
        if self._obs_spec.cells is not None:
            for ix, iy in self._obs_spec.cells:
                if 0 <= ix < self.size[0] and 0 <= iy < self.size[1]:
                    self._static_occ[ix, iy] = True
            return
        if self._obs_spec.type == "random":
            n = self._obs_spec.count
            tries = 0
            placed = 0
            # keep a halo around start and goal so the drone (with radius)
            # is not spawned immediately next to an obstacle.
            forbidden: set[tuple[int, int]] = set()
            for anchor in (self._cell(self._start), self._cell(self._goal)):
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        forbidden.add((anchor[0] + dx, anchor[1] + dy))
            while placed < n and tries < n * 20:
                ix = int(self._rng.integers(0, self.size[0]))
                iy = int(self._rng.integers(0, self.size[1]))
                if (ix, iy) in forbidden or self._static_occ[ix, iy]:
                    tries += 1
                    continue
                self._static_occ[ix, iy] = True
                placed += 1
                tries += 1
        elif self._obs_spec.type == "none":
            pass
        else:
            raise ValueError(f"unknown obstacle type: {self._obs_spec.type}")

    def _cell(self, p: np.ndarray | tuple[float, float]) -> tuple[int, int]:
        p = np.asarray(p, dtype=float)
        ix = int(np.clip(p[0] / self.resolution, 0, self.size[0] - 1))
        iy = int(np.clip(p[1] / self.resolution, 0, self.size[1] - 1))
        return ix, iy

    def is_collision(self, position: np.ndarray, radius: float) -> bool:
        # Out-of-bounds counts as collision (drone left the world).
        x, y = float(position[0]), float(position[1])
        if x < 0 or y < 0:
            return True
        if x > self.size[0] * self.resolution or y > self.size[1] * self.resolution:
            return True
        # static cells under or near the drone
        cx = int(x / self.resolution)
        cy = int(y / self.resolution)
        cells_to_check = max(1, int(np.ceil(radius / self.resolution)))
        for dx in range(-cells_to_check, cells_to_check + 1):
            for dy in range(-cells_to_check, cells_to_check + 1):
                ix, iy = cx + dx, cy + dy
                if not (0 <= ix < self.size[0] and 0 <= iy < self.size[1]):
                    continue
                if not self._static_occ[ix, iy]:
                    continue
                cell_cx = (ix + 0.5) * self.resolution
                cell_cy = (iy + 0.5) * self.resolution
                ddx = max(abs(x - cell_cx) - self.resolution / 2, 0.0)
                ddy = max(abs(y - cell_cy) - self.resolution / 2, 0.0)
                if ddx * ddx + ddy * ddy <= radius * radius:
                    return True
        # dynamic obstacles: simple sphere-sphere distance test on true positions
        for d in self._dynamic:
            sep = (d.pos[0] - x) ** 2 + (d.pos[1] - y) ** 2
            r = d.radius + radius
            if sep <= r * r:
                return True
        return False

    @property
    def start(self) -> np.ndarray:
        return self._start.copy()

    @property
    def goal(self) -> np.ndarray:
        return self._goal.copy()

    @property
    def dynamic_obstacles(self) -> list[dict]:
        return [
            {
                "position": [float(v) for v in d.pos],
                "velocity": [float(v) for v in d.vel],
                "radius": float(d.radius),
            }
            for d in self._dynamic
        ]
