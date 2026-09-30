"""Pose-estimation repeatability figure (paper appendix): FoundationPose run 47
times on a fixed fan-cover part, coarse (~30 cm) and refined (~10 cm) estimates.

Panels
  trans_xy:  x/y translation error of refined (colour = z error) and coarse estimates
  rot_rp:    roll/pitch error of refined (colour = yaw error) and coarse estimates
  drift_xy:  per-episode correction coarse -> refined in x/y (colour = drift magnitude)
  drift_rot: histogram of the per-episode rotation correction coarse -> refined

Errors are relative to the pseudo ground truth of each estimate type (component-
wise median over all episodes), as computed by the analyzer on the robot
workstation. Data: data/pe_siemens_pe_only_20260528_085358/pe_samples.csv,
written by analyze_pe_repeatability.py (copied next to it) from
pe_viz/siemens_pe_only_20260528_085358 on gabor@tueilsy-st-019.
Drawn at paper scale, see PAPER_RC in plot_forces.py.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from plot_forces import COLUMN_WIDTH_IN, PAPER_RC, TEXT_WIDTH_IN

REPO_ROOT = Path(__file__).parent
DATA_PATH = REPO_ROOT / "data" / "pe_siemens_pe_only_20260528_085358" / "pe_samples.csv"
OUTPUT_PATHS = {
    "trans_xy": REPO_ROOT / "pe_trans_xy.pdf",
    "rot_rp": REPO_ROOT / "pe_rot_roll_pitch.pdf",
    "drift_xy": REPO_ROOT / "pe_drift_trans_xy.pdf",
    "drift_rot": REPO_ROOT / "pe_drift_rot.pdf",
    "errors_1col": REPO_ROOT / "pe_errors_1col.pdf",
}

# Each panel fills just under half the text width (2x2 grid in a figure*).
PANEL_SIZE_IN = (0.48 * TEXT_WIDTH_IN, 2.35)
CMAP = "magma"
TRANS_COLS = ("trans_err_x_mm", "trans_err_y_mm", "trans_err_z_mm")
ROT_COLS = ("rpy_err_roll_deg", "rpy_err_pitch_deg", "rpy_err_yaw_deg")


def load_samples(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Return {"coarse"|"refined": {"trans": (N, 3) mm, "rot": (N, 3) deg}},
    rows aligned by episode."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = {}
    for which in ("coarse", "refined"):
        sel = sorted((r for r in rows if r["which"] == which), key=lambda r: int(r["ep"]))
        out[which] = {
            "ep": np.array([int(r["ep"]) for r in sel]),
            "trans": np.array([[float(r[c]) for c in TRANS_COLS] for r in sel]),
            "rot": np.array([[float(r[c]) for c in ROT_COLS] for r in sel]),
        }
    assert (out["coarse"]["ep"] == out["refined"]["ep"]).all()
    return out


def _style(ax, lim, xlabel: str, ylabel: str, equal: bool = True) -> None:
    """lim: half-width of a symmetric square window, or ((x0, x1), (y0, y1)).
    equal=False keeps a square box but scales x and y independently."""
    ax.axhline(0, color="grey", linewidth=0.4, linestyle="--", zorder=0)
    ax.axvline(0, color="grey", linewidth=0.4, linestyle="--", zorder=0)
    xlim, ylim = ((-lim, lim), (-lim, lim)) if np.isscalar(lim) else lim
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if equal:
        ax.set_aspect("equal")
    else:
        ax.set_box_aspect(1)
    ax.set_xlabel(xlabel, labelpad=1.5)
    ax.set_ylabel(ylabel, labelpad=1.5)
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax.tick_params(length=2, pad=1.5)


def _reference(ax) -> None:
    ax.scatter([0], [0], marker="+", s=40, color="red", linewidths=0.9, zorder=5,
               label="Ground truth")


def _colorbar(fig, mappable, ax, label: str, cax=None, fontsize=None) -> None:
    if cax is None:
        cb = fig.colorbar(mappable, ax=ax, fraction=0.05, pad=0.03)
    else:
        cb = fig.colorbar(mappable, cax=cax)
    cb.set_label(label, labelpad=2, fontsize=fontsize)
    cb.ax.tick_params(length=2, pad=1.5, labelsize=fontsize)
    cb.outline.set_linewidth(0.5)


def _draw_scatter(fig, ax, samples, key: str, lim, labels: tuple[str, str, str],
                  legend: bool = True, marker_scale: float = 1.0, equal: bool = True,
                  cax=None, small_fontsize=None):
    coarse, refined = samples["coarse"][key], samples["refined"][key]
    ax.scatter(coarse[:, 0], coarse[:, 1], s=9 * marker_scale, facecolors="none",
               edgecolors="grey", linewidths=0.5, label="Coarse", zorder=2)
    sc = ax.scatter(refined[:, 0], refined[:, 1], c=refined[:, 2], cmap=CMAP,
                    s=12 * marker_scale, edgecolors="black", linewidths=0.3,
                    label="Refined", zorder=3)
    _reference(ax)
    _style(ax, lim, labels[0], labels[1], equal=equal)
    _colorbar(fig, sc, ax, labels[2], cax=cax, fontsize=small_fontsize)
    if small_fontsize is not None:
        ax.tick_params(labelsize=small_fontsize)
    if legend:
        ax.legend(loc="upper right", fontsize=6, handletextpad=0.3, borderpad=0.3,
                  labelspacing=0.25, framealpha=0.9).get_frame().set_linewidth(0.4)


def _scatter_panel(samples, key: str, lim: float, labels: tuple[str, str, str], out: Path) -> None:
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=PANEL_SIZE_IN)
        _draw_scatter(fig, ax, samples, key, lim, labels)
        fig.tight_layout(pad=0.2)
        fig.savefig(out)
        plt.close(fig)
    print(f"[saved] {out}")


# Single-column layout, in inches: square plot boxes with the colour bar exactly as
# tall as the box; all text (ticks, axis/colour-bar labels, legend) at one small size.
_1COL = dict(height=1.3, top=0.12, bottom=0.25, left=0.31, cbar_gap=0.04,
             cbar_w=0.06, small_fontsize=6.5)


def plot_errors_single_column(samples, out: Path) -> None:
    """Translation (left) and rotation (right) error side by side, one column wide.
    Axis ranges cover every coarse and refined sample; all text at one small size."""
    L = _1COL
    W, H = COLUMN_WIDTH_IN, L["height"]
    box = H - L["top"] - L["bottom"]
    panel_w = W / 2
    with plt.rc_context(PAPER_RC):
        fig = plt.figure(figsize=(W, H))
        axes, caxes = [], []
        for i in range(2):
            x0 = i * panel_w + L["left"]
            axes.append(fig.add_axes([x0 / W, L["bottom"] / H, box / W, box / H]))
            caxes.append(fig.add_axes([(x0 + box + L["cbar_gap"]) / W, L["bottom"] / H,
                                       L["cbar_w"] / W, box / H]))
        ax_t, ax_r = axes
        _draw_scatter(fig, ax_t, samples, "trans", ((-2.9, 1.4), (-1.95, 4.15)),
                      ("$x$ error [mm]", "$y$ error [mm]", "$z$ error [mm]"),
                      legend=False, marker_scale=0.5, equal=False, cax=caxes[0],
                      small_fontsize=L["small_fontsize"])
        _draw_scatter(fig, ax_r, samples, "rot", ((-7.3, 3.2), (-4.9, 3.4)),
                      ("Roll error [deg]", "Pitch error [deg]", "Yaw error [deg]"),
                      legend=False, marker_scale=0.5, equal=False, cax=caxes[1],
                      small_fontsize=L["small_fontsize"])
        for ax in axes:
            ax.set_box_aspect(None)  # the box is already square via its position
            ax.xaxis.labelpad = 1.0
            ax.yaxis.labelpad = 1.0
            ax.xaxis.label.set_size(L["small_fontsize"])
            ax.yaxis.label.set_size(L["small_fontsize"])
        ax_t.set_xticks([-2, -1, 0, 1])
        ax_t.set_yticks([-1, 0, 1, 2, 3, 4])
        ax_r.set_xticks([-6, -3, 0, 3])
        ax_r.set_yticks([-4, -2, 0, 2])
        handles, names = ax_t.get_legend_handles_labels()
        fig.legend(handles, names, loc="upper center", ncol=3, handletextpad=0.3,
                   columnspacing=1.2, borderpad=0.0, borderaxespad=0.0, frameon=False,
                   bbox_to_anchor=(0.5, 1.0), fontsize=L["small_fontsize"])
        fig.savefig(out)
        plt.close(fig)
    print(f"[saved] {out}")


def plot_drift_xy(samples, out: Path) -> None:
    c, r = samples["coarse"]["trans"][:, :2], samples["refined"]["trans"][:, :2]
    mag = np.linalg.norm(c - r, axis=1)
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=PANEL_SIZE_IN)
        lc = LineCollection(np.stack([c, r], axis=1), array=mag, cmap=CMAP,
                            linewidths=0.6, zorder=2)
        ax.add_collection(lc)
        ax.scatter(c[:, 0], c[:, 1], s=6, facecolors="none", edgecolors="grey",
                   linewidths=0.5, label="Coarse", zorder=3)
        ax.scatter(r[:, 0], r[:, 1], s=4, color="black", label="Refined", zorder=4)
        _reference(ax)
        _style(ax, ((-3.0, 4.0), (-2.0, 4.2)), "$x$ error [mm]", "$y$ error [mm]")
        _colorbar(fig, lc, ax, "Correction [mm]")
        ax.legend(loc="lower right", fontsize=6, handletextpad=0.3, borderpad=0.3,
                  labelspacing=0.25, framealpha=0.9).get_frame().set_linewidth(0.4)
        fig.tight_layout(pad=0.2)
        fig.savefig(out)
        plt.close(fig)
    print(f"[saved] {out}")


def plot_drift_rot(samples, out: Path) -> None:
    # Norm of the per-axis (roll, pitch, yaw) correction, as in the original figure.
    drift = np.linalg.norm(samples["coarse"]["rot"] - samples["refined"]["rot"], axis=1)
    mean, std = drift.mean(), drift.std()
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=PANEL_SIZE_IN)
        counts, edges, patches = ax.hist(drift, bins=np.arange(0, 7.5, 0.25),
                                         edgecolor="black", linewidth=0.3)
        cmap = plt.get_cmap(CMAP)
        centers = 0.5 * (edges[:-1] + edges[1:])
        for p, x in zip(patches, centers):
            p.set_facecolor(cmap(0.1 + 0.8 * x / edges[-1]))
        ax.axvline(mean, color="red", linewidth=0.8, label=f"Mean: {mean:.2f}$^\\circ$")
        for s in (-1, 1):
            ax.axvline(mean + s * std, color="orange", linewidth=0.7, linestyle="--",
                       label=f"$\\pm1\\sigma$: {std:.2f}$^\\circ$" if s > 0 else None)
        ax.set_xlim(0, edges[-1])
        ax.set_xlabel("Rotation correction [deg]", labelpad=1.5)
        ax.set_ylabel("Count", labelpad=1.5)
        ax.grid(True, alpha=0.3, linewidth=0.4)
        ax.tick_params(length=2, pad=1.5)
        ax.legend(loc="upper right", fontsize=6, handletextpad=0.3, borderpad=0.3,
                  labelspacing=0.25, framealpha=0.9).get_frame().set_linewidth(0.4)
        fig.tight_layout(pad=0.2)
        fig.savefig(out)
        plt.close(fig)
    print(f"[saved] {out}  (n={len(drift)}, mean={mean:.3f}, std={std:.3f})")


def main() -> None:
    samples = load_samples(DATA_PATH)
    print(f"[loaded] {len(samples['refined']['ep'])} episodes from {DATA_PATH}")
    _scatter_panel(samples, "trans", 1.25,
                   ("$x$ error [mm]", "$y$ error [mm]", "$z$ error [mm]"), OUTPUT_PATHS["trans_xy"])
    _scatter_panel(samples, "rot", 7.5,
                   ("Roll error [deg]", "Pitch error [deg]", "Yaw error [deg]"), OUTPUT_PATHS["rot_rp"])
    plot_drift_xy(samples, OUTPUT_PATHS["drift_xy"])
    plot_drift_rot(samples, OUTPUT_PATHS["drift_rot"])
    plot_errors_single_column(samples, OUTPUT_PATHS["errors_1col"])


if __name__ == "__main__":
    main()
