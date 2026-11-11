"""Figures: docs/figure.png (1600x900, README) and docs/post-figure.png (960x700, dark, phone-sized)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

from .terrain import Grid, hillshade  # noqa: E402

DARK = {
    "bg": "#16213A", "grid": "#2A3750", "text": "#F8FAFC", "muted": "#A9B6CB", "accent": "#38BDF8",
    "green": "#34D399", "amber": "#FBBF24", "red": "#F87171", "grey": "#64748B", "violet": "#A78BFA",
}
FONT = ["Segoe UI", "DejaVu Sans"]
CMAP = LinearSegmentedColormap.from_list("prio", ["#64748B", "#38BDF8", "#FBBF24", "#F87171"])
SOURCE = "Terrain: swissALTIRegio, source: swisstopo. Line: railway network, source: FOT. Wind/snow: MeteoSwiss 2020-2025."


def _line_collection(sc, col, lw, norm):
    pts = np.column_stack([sc["e"], sc["n"]]) / 1000.0
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    vals = sc[col].to_numpy(float)
    lc = LineCollection(segs, cmap=CMAP, norm=norm, linewidths=lw, capstyle="round")
    lc.set_array((vals[:-1] + vals[1:]) / 2)
    return lc


def _map(ax, dem: Grid, sc, top, col, lw, label_color, halo, fontsize, n_labels, shade_alpha=1.0, dark=False):
    w, s, e, n = dem.bounds
    hs = hillshade(dem.z, dem.res)
    cmap = "gray" if not dark else LinearSegmentedColormap.from_list("hs", ["#0B1222", "#5C6B86"])
    ax.imshow(hs, extent=(w / 1000, e / 1000, s / 1000, n / 1000), cmap=cmap, alpha=shade_alpha, interpolation="bilinear")
    norm = Normalize(0, 100)
    ax.add_collection(_line_collection(sc, col, lw, norm))
    labels = top.head(n_labels).sort_values("km_from").reset_index(drop=True)
    for j, r in labels.iterrows():
        x, y = r["e"] / 1000, r["n"] / 1000
        dy = 0.75 if j % 2 == 0 else -0.75  # alternate above/below the line so close neighbours stay readable
        ax.plot(x, y, "o", ms=4, color=label_color if not dark else halo, zorder=5)
        ax.annotate(str(int(r["rank"])), (x, y), xytext=(x, y + dy), ha="center", va="center", fontsize=fontsize * 0.8,
                    color=label_color, fontweight="bold", zorder=6,
                    bbox={"boxstyle": "circle,pad=0.25", "fc": halo, "ec": label_color, "lw": 1.2},
                    arrowprops={"arrowstyle": "-", "color": halo if dark else label_color, "lw": 1.0})
    pad = 0.6
    ax.set_xlim(sc["e"].min() / 1000 - pad, sc["e"].max() / 1000 + pad)
    ax.set_ylim(sc["n"].min() / 1000 - pad * 1.6, sc["n"].max() / 1000 + pad * 1.6)
    ax.set_aspect("equal")
    return norm


def figure(sc, top, dem: Grid, val: dict, path: Path, ref_name: str) -> None:
    plt.rcParams.update({"font.family": FONT, "font.size": 12})
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1], width_ratios=[1, 1, 0.82], hspace=0.3, wspace=0.3,
                          left=0.05, right=0.95, top=0.88, bottom=0.08)
    fig.suptitle("Where a 28.6 km mountain rail line is most exposed to wind and drifting snow (100 m sections)",
                 x=0.05, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.05, 0.925, SOURCE, fontsize=10.5, color="#555")
    ax = fig.add_subplot(gs[0, :2])
    norm = _map(ax, dem, sc, top, "priority", 5, "#111", "white", 11, 10, shade_alpha=0.85)
    ax.set_xlabel("LV95 east (km)")
    ax.set_ylabel("north (km)")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("priority (0-100)")
    ax.set_title("Priority = mean percentile of crosswind storm exposure and drift index; numbers = top 10", fontsize=12,
                 loc="left")

    km = (sc["km_from"] + sc["km_to"]) / 2
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(km, sc["wind_score"], color="#0369A1", lw=1.6, label="crosswind storm exposure (m/s)")
    ax2.set_ylabel("crosswind exposure (m/s)", color="#0369A1")
    ax2.set_xlabel("distance along line (km)")
    ax3 = ax2.twinx()
    ax3.fill_between(km, sc["drift_index"], color="#F59E0B", alpha=0.35, lw=0, label="drift index (0-100)")
    ax3.set_ylabel("drift index (0-100)", color="#B45309")
    ax3.set_ylim(0, 105)
    for _, r in top.iterrows():
        ax2.axvline((r["km_from"] + r["km_to"]) / 2, color="#DC2626", lw=0.8, alpha=0.5)
    ax2.set_title("Along the line: crosswind storm exposure (blue) and drift index (orange); red = top 10", fontsize=12,
                  loc="left")
    ax2.set_xlim(km.min(), km.max())
    ax2.grid(alpha=0.25)

    ax4 = fig.add_subplot(gs[0, 2])
    pred = val["predictions"]
    met = val["models"].set_index("model")
    lim = (0, float(np.nanmax([pred["obs"].max(), pred["elevation+sx300"].max()])) * 1.08)
    ax4.plot(lim, lim, color="#999", lw=1)
    ax4.scatter(pred["obs"], pred["elevation"], s=16, color="#9CA3AF",
                label=f"elevation only: RMSE {met.loc['elevation', 'loo_rmse']:.2f}")
    ax4.scatter(pred["obs"], pred["elevation+sx300"], s=16, color="#0369A1",
                label=f"elevation + Sx: RMSE {met.loc['elevation+sx300', 'loo_rmse']:.2f}")
    ax4.set_xlim(lim)
    ax4.set_ylim(lim)
    ax4.set_xlabel("observed winter mean wind (m/s)")
    ax4.set_ylabel("predicted, station left out (m/s)")
    ax4.set_title(f"Validation: {val['n_stations']} weather stations, leave-one-out", fontsize=12, loc="left")
    ax4.legend(fontsize=10, loc="upper left", frameon=False)
    ax4.grid(alpha=0.25)
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def post_figure(sc, top, dem: Grid, summary: dict, path: Path) -> None:
    c = DARK
    plt.rcParams.update({"font.family": FONT, "font.size": 12, "text.color": c["text"], "axes.labelcolor": c["muted"],
                         "xtick.color": c["muted"], "ytick.color": c["muted"], "axes.edgecolor": c["grid"]})
    fig = plt.figure(figsize=(9.6, 7.0), dpi=100, facecolor=c["bg"])
    fig.text(0.04, 0.955, "Which 100 m of a mountain line catch the most wind and drift", fontsize=14, fontweight="bold",
             color=c["text"], va="top")
    fig.text(0.04, 0.905, "Open data: swisstopo terrain, MeteoSwiss stations 2020-2025, federal rail network",
             fontsize=11.5, color=c["muted"], va="top")
    ax = fig.add_axes([0.04, 0.36, 0.92, 0.5], facecolor=c["bg"])
    _map(ax, dem, sc, top, "priority", 6, c["bg"], c["amber"], 12, 5, dark=True)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(0.01, 0.03, "colour = priority (blue low, amber/red high); numbers = top 5", transform=ax.transAxes,
            fontsize=11.5, color=c["text"])
    ax2 = fig.add_axes([0.1, 0.09, 0.86, 0.22], facecolor=c["bg"])
    km = (sc["km_from"] + sc["km_to"]) / 2
    ax2.fill_between(km, sc["priority"], color=c["accent"], alpha=0.25, lw=0)
    ax2.plot(km, sc["priority"], color=c["accent"], lw=1.8)
    for _, r in top.head(5).iterrows():
        x = (r["km_from"] + r["km_to"]) / 2
        ax2.plot(x, r["priority"], "o", ms=13, mfc=c["amber"], mec=c["bg"])
        ax2.text(x, r["priority"], str(int(r["rank"])), ha="center", va="center", fontsize=10, color=c["bg"], fontweight="bold")
    ax2.set_xlim(km.min(), km.max())
    ax2.set_ylim(0, 112)
    ax2.set_xlabel("km along the line", fontsize=11.5)
    ax2.set_ylabel("priority", fontsize=11.5)
    ax2.grid(color=c["grid"], lw=0.8)
    ax2.tick_params(labelsize=11)
    for sp in ax2.spines.values():
        sp.set_color(c["grid"])
    fig.savefig(path, dpi=100, facecolor=c["bg"])
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)
