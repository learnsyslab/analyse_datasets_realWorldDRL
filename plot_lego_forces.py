"""Boxplots of per-episode peak force and peak torque for the Siemens task,
one box per policy.

Episode-level statistic = peak |F| (or peak |τ|) over the episode's frames.
For the baseline policies this is computed from the data parquets; for "Ours"
it is read from the eval results JSON (force_N.max / torque_Nm.max per
episode). Datasets are downloaded in full into ./datasets/<repo_id>/.
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": plt.rcParams["font.size"] * 5})
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars

import config

disable_progress_bars()
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

REPO_ROOT = Path(__file__).parent
DATASETS_DIR = REPO_ROOT / "datasets"
FORCE_OUTPUT_PATH = REPO_ROOT / "siemens_forces_boxplot.pdf"
TORQUE_OUTPUT_PATH = REPO_ROOT / "siemens_torques_boxplot.pdf"

FT_COL = "observation.state.sensors_bota_ft_sensor"

# FT sensor is 6-dim: [Fx, Fy, Fz, Tx, Ty, Tz].
METRICS = {
    "force": {
        "slice": slice(0, 3),
        "json_key": "force_N",
        "ylabel": "Episode max. F [N]",
        "output": FORCE_OUTPUT_PATH,
    },
    "torque": {
        "slice": slice(3, 6),
        "json_key": "torque_Nm",
        "ylabel": "Episode max. T [Nm]",
        "output": TORQUE_OUTPUT_PATH,
    },
}

POLICY_LABELS = {
    "Pi05": "Finet.\nPi05",
    "Ours (dataset)": "Ours\n(learned\npolicy)",
    "Ditflow Novice": "Ditflow\n(novice)",
    "Diffusion": "DP",
}
OURS_JSON_LABEL = "Ours\n(push\ndown)"

# Baseline policy label -> task instance. Order here defines the x-axis order;
# "Ours" is appended last from its eval results JSON (see OURS_JSON below).
OURS_TASK = config.OursSiemensGeneralization1()
POLICIES = {
    "Diffusion": config.SiemensDiffusion(),
    "Ditflow": config.SiemensDitflow(),
    "Ditflow Novice": config.LegoSimpleDitflowNovice(),
    "Pi05": config.SiemensPi05(),
    "Ours (dataset)": OURS_TASK,
}

OURS_JSON = DATASETS_DIR / OURS_TASK.datasets[0] / "eval_results.json"


def episode_peak_forces(repo_id: str, axes: slice = slice(0, 3)) -> list[float]:
    """Download a dataset and return the per-episode peak magnitude of the
    FT-sensor sub-vector selected by ``axes`` (default: force = first 3 axes)."""
    local_dir = DATASETS_DIR / repo_id
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(local_dir))

    data_files = sorted((local_dir / "data").rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No data parquet files found under {local_dir}")

    peak: dict[int, float] = {}
    for p in data_files:
        df = pd.read_parquet(p, columns=["episode_index", FT_COL])
        if df.empty:
            continue
        vecs = np.stack(list(df[FT_COL].to_numpy()))
        mag = np.linalg.norm(vecs[:, axes], axis=1)
        ep_idx = df["episode_index"].to_numpy()
        for ep in np.unique(ep_idx):
            m = float(mag[ep_idx == ep].max())
            ep_int = int(ep)
            peak[ep_int] = max(peak.get(ep_int, m), m)

    return list(peak.values())


def episode_peak_forces_from_json(json_path: Path) -> list[float]:
    """Return the per-episode max force (force_N.max) from an eval results JSON."""
    with open(json_path) as f:
        results = json.load(f)
    return [ep["force_N"]["max"] for ep in results["episodes"]]


def episode_peaks_from_json(json_path: Path, key: str) -> list[float]:
    """Return the per-episode max of ``ep[key]['max']`` from an eval results JSON."""
    with open(json_path) as f:
        results = json.load(f)
    return [ep[key]["max"] for ep in results["episodes"]]


def plot_metric(metric_name: str) -> None:
    cfg = METRICS[metric_name]
    data: list[list[float]] = []
    labels: list[str] = []

    for name, task in POLICIES.items():
        print(f"  {name}...", flush=True)
        peaks: list[float] = []
        for repo_id in task.datasets:
            peaks.extend(episode_peak_forces(repo_id, axes=cfg["slice"]))
        data.append(peaks)
        labels.append(POLICY_LABELS.get(name, name))

    print(f"  {OURS_JSON_LABEL}...", flush=True)
    ours = episode_peaks_from_json(OURS_JSON, cfg["json_key"])
    data.append(ours)
    labels.append(OURS_JSON_LABEL)

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_ylabel(cfg["ylabel"])
    ax.set_title("Fan Cover")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(cfg["output"])
    plt.close(fig)
    print(f"Wrote plot to {cfg['output']}")


def main() -> None:
    for metric_name in METRICS:
        print(f"=== {metric_name} ===")
        plot_metric(metric_name)


if __name__ == "__main__":
    main()
