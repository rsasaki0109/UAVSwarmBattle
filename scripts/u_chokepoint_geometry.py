"""U step 2: visual side-by-side of intersection_chokepoint (uniform
wins 95%) vs city_chokepoint (uniform loses, argmin wins 45%). Both
have 4 cubes; only city has corner buildings creating a narrow
corridor. The mechanism: forced-commitment gap width < uncertainty
radius → argmin > uniform.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import yaml

OUT = Path("docs/images/u_chokepoint_geometry.png")

CELLS = [
    ("intersection_chokepoint",
     "examples/exp_intersection_chokepoint_noisy30_warmup_select_mppi_n20.yaml",
     "uniform t=10 WINS 95%  (N+P hit)"),
    ("city_chokepoint",
     "examples/exp_city_chokepoint_noisy30_warmup_select_mppi_n20.yaml",
     "uniform t=10 LOSES 35%, argmin t=0.1 WINS 45%  (N+P MISS)"),
]


def main() -> int:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, (tag, ypath, subtitle) in zip(axes, CELLS):
        raw = yaml.safe_load(open(ypath))
        sc = raw["scenario"]
        size = sc["size"]
        boxes = sc["obstacles"].get("boxes", [])
        drones = sc["drones"]
        dyn = sc.get("dynamic_obstacles", [])

        # World outline
        ax.add_patch(patches.Rectangle((0, 0), size[0], size[1],
                                        fill=False, edgecolor="black", lw=1.0))
        # Buildings/cubes
        for b in boxes:
            cx, cy, _ = b["center"]
            sx, sy, _ = b["size"]
            kind_color = "#8c564b" if sx >= 10 else "#ff7f0e"  # building vs cube
            kind_label = "building" if sx >= 10 else "cube"
            ax.add_patch(patches.Rectangle(
                (cx - sx / 2, cy - sy / 2), sx, sy,
                facecolor=kind_color, alpha=0.55, edgecolor="black", lw=0.7,
            ))
        # Drones start/goal
        for i, d in enumerate(drones):
            sx, sy, _ = d["start"]
            gx, gy, _ = d["goal"]
            color = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"][i % 4]
            ax.plot([sx, gx], [sy, gy], "--", color=color, lw=1.2, alpha=0.6)
            ax.scatter([sx], [sy], s=70, color=color, marker="o",
                       edgecolor="black", zorder=5, label=f"{d['name']} start")
            ax.scatter([gx], [gy], s=110, color=color, marker="*",
                       edgecolor="black", zorder=5)
        # Dynamic intruder(s)
        for k, do in enumerate(dyn):
            sx, sy, _ = do["start"]
            ax.scatter([sx], [sy], s=140, color="#7f7f7f", marker="X",
                       edgecolor="black", zorder=6,
                       label="intruder" if k == 0 else None)

        # Corridor width annotation for city case
        if tag == "city_chokepoint":
            # Corridor between buildings is x in [24, 36], y in [24, 36]
            # Cube cluster at x in [24, 36], y in [24, 36] (cubes 4x4 at corners of 8x8)
            # Gap between west building and west cubes: from x=24 (east edge of west bldg)
            # to x=24 (west edge of west cube at center 26). Gap = 0! Actually:
            # West bldg: center 12, size 24 → x in [0, 24]
            # West cube: center 26, size 4 → x in [24, 28]
            # So they TOUCH. The clear gap is between the two west cubes
            # along y: cube at (26,26) y in [24,28], cube at (26,34) y in [32,36].
            # Vertical gap between them along y=30 is 32-28=4m wide.
            ax.annotate("", xy=(26, 32), xytext=(26, 28),
                        arrowprops=dict(arrowstyle="<->", color="red", lw=2))
            ax.text(22, 30, "4 m\nforced\ngap", color="red", fontsize=9,
                    ha="right", fontweight="bold")
            ax.annotate("12 m corridor\n(walls)", xy=(48, 30), xytext=(43, 12),
                        fontsize=8, color="#8c564b",
                        arrowprops=dict(arrowstyle="->", color="#8c564b", lw=1.0))

        ax.set_xlim(-2, size[0] + 2)
        ax.set_ylim(-2, size[1] + 2)
        ax.set_aspect("equal")
        ax.set_title(f"{tag}  ({size[0]}x{size[1]} m)\n{subtitle}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        if tag == "intersection_chokepoint":
            ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "U mechanism: same 4 cubes + intruder, but corner buildings turn an open chokepoint "
        "into a narrow squeeze. Forced gap (4 m) < uncertainty radius → argmin > uniform.",
        fontsize=11,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
