"""HIL-SERL training figure: human intervention rate over training episodes
for the Lego insertion phase.

Data: data/hilserl_lego/episode_stat.csv, one row per training episode
(intervention_steps, intervention_count, return, episode_length). The rate of a
bin is the share of steps under human takeover, pooled over BIN episodes.
Drawn at paper scale, see PAPER_RC in plot_forces.py.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_forces import COLUMN_WIDTH_IN, PAPER_RC

REPO_ROOT = Path(__file__).parent
DATA_PATH = REPO_ROOT / "data" / "hilserl_lego" / "episode_stat.csv"
OUTPUT_PATH = REPO_ROOT / "hilserl_intervention_rate.pdf"

BIN = 10  # episodes per point
# The caption sits beside the plot, so it takes only part of the column (matches the
# minipage width in main.tex).
WIDTH_FRACTION = 0.56


def intervention_rate(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    bins = np.arange(len(df)) // BIN
    grouped = df.groupby(bins)
    rate = 100 * grouped["intervention_steps"].sum() / grouped["episode_length"].sum()
    return rate.index.to_numpy() * BIN, rate.to_numpy()


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"{len(df)} episodes, {df['intervention_count'].sum()} takeovers in "
          f"{(df['intervention_count'] > 0).sum()} episodes")
    x, rate = intervention_rate(df)

    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(WIDTH_FRACTION * COLUMN_WIDTH_IN, 0.88))
        ax.plot(x, rate, color="C0", linewidth=0.9)
        ax.fill_between(x, rate, color="C0", alpha=0.15, linewidth=0)
        ax.set_xlim(0, len(df))
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 50, 100])
        ax.set_xticks([0, 100, 200, 300])
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Intervention\nrate [%]")
        ax.grid(True, alpha=0.3, linewidth=0.4)
        ax.tick_params(length=2, pad=1.5)
        fig.tight_layout(pad=0.2)
        fig.savefig(OUTPUT_PATH)
        plt.close(fig)
    print(f"Wrote plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
