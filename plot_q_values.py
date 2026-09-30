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

from plot_forces import PAPER_RC, TEXT_WIDTH_IN

REPO_ROOT = Path(__file__).parent
DATA_PATH = REPO_ROOT / "data" / "q_values_run_s_1b_0.82" / "state_values_raw.npz"
OUTPUT_PATH = REPO_ROOT / "q_values_success_classifier.pdf"

EPISODE_LEN = 150  # truncation length; successful episodes are aligned to end here
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


def main() -> None:
    success, truncated, meta = load_values(DATA_PATH)
    print(f"{len(truncated)} truncated, {len(success)} successful episodes "
          f"({meta['checkpoint_dir']}, {meta['dataset_dir']})")

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
        for ax in axes:
            ax.axhline(R_SUCCESS, color="grey", linewidth=0.5, linestyle="--", zorder=0)
            ax.set_xlim(0, EPISODE_LEN)
            ax.set_ylim(-0.5, 10.5)
            ax.set_yticks([0, 2, 4, 6, 8, 10])
            ax.grid(True, alpha=0.3, linewidth=0.4)
            ax.tick_params(length=2, pad=1.5)
        ax_trunc.text(EPISODE_LEN - 2, R_SUCCESS, r"$R_\mathrm{success}$", ha="right",
                      va="top", color="grey", fontsize=7)

        fig.tight_layout(pad=0.2, w_pad=0.6)
        fig.savefig(OUTPUT_PATH)
        plt.close(fig)
    print(f"Wrote plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
