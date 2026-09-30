"""Q-function success classifier figure: per-episode state values
V(s_t) = mean_i Q_i(s_t, pi(s_t)) for truncated (left) and successful,
end-aligned (right) training rollouts.

Data: data/q_values_run_s_1b_0.82/state_values_raw.npz, exported on the robot
workstation with data/q_values_run_s_1b_0.82/export_state_values.py from the
checkpoint rw_1cft_d300vs1b/pretrain_5 and the dataset run_s_1b_0.82 (see the
"meta" entry in the npz). Drawn at paper scale, see PAPER_RC in plot_forces.py.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_forces import COLUMN_WIDTH_IN, PAPER_RC, TEXT_WIDTH_IN

REPO_ROOT = Path(__file__).parent
DATA_PATH = REPO_ROOT / "data" / "q_values_run_s_1b_0.82" / "state_values_raw.npz"
OUTPUT_PATH = REPO_ROOT / "q_values_success_classifier.pdf"
OUTPUT_PATH_1COL_H = REPO_ROOT / "q_values_success_classifier_1col_h.pdf"
OUTPUT_PATH_1COL_V = REPO_ROOT / "q_values_success_classifier_1col_v.pdf"

EPISODE_LEN = 150  # truncation length; successful episodes are aligned to end here
X_MAX = 160        # a bit past the end so it is visible that all rollouts stop at 150
R_SUCCESS = 10.0   # sparse terminal reward = max Q-value


def load_values(path: Path) -> tuple[list[np.ndarray], list[np.ndarray], dict]:
    d = np.load(path)
    meta = json.loads(str(d["meta"]))
    success = [d[f"success_{i}"] for i in range(meta["n_success"])]
    truncated = [d[f"truncated_{i}"] for i in range(meta["n_truncated"])]
    return success, truncated, meta


def _plot_lines(ax, series: list[np.ndarray], end_aligned: bool) -> None:
    for values in series:
        if end_aligned:
            x = np.arange(EPISODE_LEN - len(values), EPISODE_LEN)
        else:
            x = np.arange(len(values))
        ax.plot(x, values, alpha=0.65, linewidth=0.5)


def _style_axes(axes, x_labels: bool = True) -> None:
    for ax in axes:
        ax.axhline(R_SUCCESS, color="grey", linewidth=0.5, linestyle="--", zorder=0)
        ax.axvline(EPISODE_LEN, color="grey", linewidth=0.5, linestyle="--", zorder=0)
        ax.set_xlim(0, X_MAX)
        ax.set_xticks(np.arange(0, EPISODE_LEN + 1, 30))
        ax.set_ylim(-0.5, 10.5)
        ax.set_yticks([0, 2, 4, 6, 8, 10])
        ax.grid(True, alpha=0.3, linewidth=0.4)
        ax.tick_params(length=2, pad=1.5)
    axes[0].text(EPISODE_LEN - 3, R_SUCCESS, r"$R_\mathrm{success}$", ha="right",
                 va="top", color="grey", fontsize=7)


def plot_full_width(truncated, success) -> None:
    """Two panels side by side across the text width."""
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 1.9), sharey=True)
        ax_trunc, ax_succ = axes
        _plot_lines(ax_trunc, truncated, end_aligned=False)
        _plot_lines(ax_succ, success, end_aligned=True)
        ax_trunc.set_title(f"Truncated rollouts (n={len(truncated)})")
        ax_succ.set_title(f"Successful rollouts, end-aligned (n={len(success)})")
        ax_trunc.set_ylabel("Mean Q-value")
        ax_trunc.set_xlabel("Timestep")
        ax_succ.set_xlabel(f"Aligned timestep (all episodes end at $t={EPISODE_LEN}$)")
        _style_axes(axes)
        fig.tight_layout(pad=0.2, w_pad=0.6)
        fig.savefig(OUTPUT_PATH)
        plt.close(fig)
    print(f"Wrote plot to {OUTPUT_PATH}")


def plot_single_column_horizontal(truncated, success) -> None:
    """Two panels side by side within one column."""
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH_IN, 1.4), sharey=True)
        ax_trunc, ax_succ = axes
        _plot_lines(ax_trunc, truncated, end_aligned=False)
        _plot_lines(ax_succ, success, end_aligned=True)
        # Panel names inside the axes (top left) instead of titles to save height.
        for ax, name in ((ax_trunc, "Truncated"), (ax_succ, "Success")):
            ax.text(0.03, 0.95, name, transform=ax.transAxes, ha="left", va="top")
        ax_trunc.set_ylabel("Mean Q-value")
        ax_trunc.set_xlabel("Timestep")
        ax_succ.set_xlabel("Aligned timestep")
        _style_axes(axes)
        for ax in axes:
            ax.set_xticks(np.arange(0, EPISODE_LEN + 1, 50))
        fig.tight_layout(pad=0.2, w_pad=0.5)
        fig.savefig(OUTPUT_PATH_1COL_H)
        plt.close(fig)
    print(f"Wrote plot to {OUTPUT_PATH_1COL_H}")


def plot_single_column_vertical(truncated, success) -> None:
    """Truncated over successful, sharing the timestep axis, within one column."""
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH_IN, 2.5), sharex=True)
        ax_trunc, ax_succ = axes
        _plot_lines(ax_trunc, truncated, end_aligned=False)
        _plot_lines(ax_succ, success, end_aligned=True)
        ax_trunc.set_title(f"Truncated rollouts (n={len(truncated)})")
        ax_succ.set_title(f"Successful rollouts, end-aligned (n={len(success)})")
        for ax in axes:
            ax.set_ylabel("Mean Q-value")
        ax_succ.set_xlabel(f"Timestep (successful rollouts aligned to end at $t={EPISODE_LEN}$)")
        _style_axes(axes)
        fig.align_ylabels(axes)
        fig.tight_layout(pad=0.2, h_pad=0.4)
        fig.savefig(OUTPUT_PATH_1COL_V)
        plt.close(fig)
    print(f"Wrote plot to {OUTPUT_PATH_1COL_V}")


def main() -> None:
    success, truncated, meta = load_values(DATA_PATH)
    print(f"{len(truncated)} truncated, {len(success)} successful episodes "
          f"({meta['checkpoint_dir']}, {meta['dataset_dir']})")
    plot_full_width(truncated, success)
    plot_single_column_horizontal(truncated, success)
    plot_single_column_vertical(truncated, success)


if __name__ == "__main__":
    main()
