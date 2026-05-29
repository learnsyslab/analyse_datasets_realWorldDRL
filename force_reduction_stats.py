"""Print mean peak-force/torque of Ours (push-down only) vs each baseline,
pooled across Fan cover + Lego + Shelf.

- Ours mean: pool the per-episode peaks from the push-down eval_results.json
  for all tasks into one set, take the mean. (Learned-policy peaks are ignored.)
- Per-baseline mean: pool the per-episode peaks of that baseline across all
  tasks where it appears.
- Overall baseline mean: pool every baseline-episode peak across all tasks.

Standalone: only depends on ``config`` and the locally-downloaded datasets
under ``./datasets/<repo_id>/``. If a dataset is missing locally it is
downloaded via ``huggingface_hub``.
"""

import json
import logging
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars

import config

disable_progress_bars()
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

REPO_ROOT = Path(__file__).parent
DATASETS_DIR = REPO_ROOT / "datasets"

FT_COL = "observation.state.sensors_bota_ft_sensor"

METRICS = {
    "force": {"slice": slice(0, 3), "json_key": "force_N", "unit": "N"},
    "torque": {"slice": slice(3, 6), "json_key": "torque_Nm", "unit": "Nm"},
}

BASELINES_LEGO = {
    "Diffusion": config.LegoSimpleDiffusion(),
    "Ditflow": config.LegoSimpleDitflow(),
    "Ditflow Novice": config.LegoSimpleDitflowNovice(),
    "Pi05": config.LegoSimplePi05(),
}
BASELINES_SIEMENS = {
    "Diffusion": config.SiemensDiffusion(),
    "Ditflow": config.SiemensDitflow(),
    "Pi05": config.SiemensPi05(),
}
BASELINES_SHELF = {
    "Diffusion": config.ShelfDiffusion(),
    "Ditflow": config.ShelfDitflow(),
    "Ditflow Novice": config.ShelfDtiflowJim(),
    "Pi05": config.ShelfPi05(),
}

TASKS = {
    "Fan cover": {
        "baselines": BASELINES_SIEMENS,
        "ours_pushdown_task": config.OursSiemensGeneralization1(),
    },
    "Lego": {
        "baselines": BASELINES_LEGO,
        "ours_pushdown_task": config.OursLegoFt20(),
    },
    "Shelf stocking": {
        "baselines": BASELINES_SHELF,
        "ours_pushdown_json": DATASETS_DIR / "stats_ours_shelf_task" / "ft_statistics.json",
    },
}


def episode_peaks_from_parquet(repo_id: str, axes: slice) -> list[float]:
    """Per-episode peak |sub-vector| of the FT sensor over ``axes``."""
    local_dir = DATASETS_DIR / repo_id
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(local_dir))
    data_files = sorted((local_dir / "data").rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet files under {local_dir}")
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


def episode_peaks_from_json(repo_id: str, json_key: str) -> list[float]:
    """Per-episode max from ``eval_results.json`` (key path: ep[json_key]['max'])."""
    json_path = DATASETS_DIR / repo_id / "eval_results.json"
    if not json_path.is_file():
        snapshot_download(
            repo_id=repo_id, repo_type="dataset", local_dir=str(DATASETS_DIR / repo_id)
        )
    with open(json_path) as f:
        results = json.load(f)
    return [ep[json_key]["max"] for ep in results["episodes"]]


def episode_peaks_from_json_path(json_path: Path, json_key: str) -> list[float]:
    """Per-episode max from a JSON file given by direct path (no HF download)."""
    with open(json_path) as f:
        results = json.load(f)
    return [ep[json_key]["max"] for ep in results["episodes"]]


def baseline_peaks(task, axes: slice) -> list[float]:
    out: list[float] = []
    for repo_id in task.datasets:
        out.extend(episode_peaks_from_parquet(repo_id, axes))
    return out


def main() -> None:
    for metric_name, cfg in METRICS.items():
        unit = cfg["unit"]
        print(f"\n========== {metric_name.upper()} ==========")

        ours_peaks_by_task: dict[str, list[float]] = {}
        baseline_peaks_by_task: dict[str, dict[str, list[float]]] = {}
        for tname, entry in TASKS.items():
            if "ours_pushdown_json" in entry:
                ours_peaks_by_task[tname] = episode_peaks_from_json_path(
                    entry["ours_pushdown_json"], cfg["json_key"]
                )
            else:
                ours_peaks_by_task[tname] = episode_peaks_from_json(
                    entry["ours_pushdown_task"].datasets[0], cfg["json_key"]
                )
            baseline_peaks_by_task[tname] = {
                bname: baseline_peaks(btask, cfg["slice"])
                for bname, btask in entry["baselines"].items()
            }

        # Per-task breakdown
        for tname in TASKS:
            ours = ours_peaks_by_task[tname]
            ours_mean = float(np.mean(ours))
            print(f"\n  --- {tname} ---")
            print(
                f"    Ours (push-down) mean episodic max {metric_name}: "
                f"{ours_mean:.2f} {unit}  (n={len(ours)})"
            )
            for bname, peaks in baseline_peaks_by_task[tname].items():
                if not peaks:
                    continue
                m = float(np.mean(peaks))
                red_abs = m - ours_mean
                red_pct = 100.0 * red_abs / m if m > 0 else float("nan")
                print(
                    f"    vs {bname:14s}: baseline mean = {m:7.2f} {unit} "
                    f"(n={len(peaks):3d})  ->  Ours lower by "
                    f"{red_abs:7.2f} {unit} ({red_pct:5.1f}%)"
                )

        # Overall, pooled across tasks
        ours_all = [v for vals in ours_peaks_by_task.values() for v in vals]
        ours_mean_all = float(np.mean(ours_all))
        baseline_pooled: dict[str, list[float]] = {}
        for per_task in baseline_peaks_by_task.values():
            for bname, peaks in per_task.items():
                baseline_pooled.setdefault(bname, []).extend(peaks)
        all_baseline_vals = [v for vs in baseline_pooled.values() for v in vs]
        all_baseline_mean = float(np.mean(all_baseline_vals))

        print("\n  --- OVERALL (Fan cover + Lego + Shelf pooled) ---")
        print(
            f"    Ours (push-down) pooled mean: {ours_mean_all:.2f} {unit}  "
            f"(n={len(ours_all)})"
        )
        for bname, peaks in baseline_pooled.items():
            m = float(np.mean(peaks))
            red_abs = m - ours_mean_all
            red_pct = 100.0 * red_abs / m if m > 0 else float("nan")
            print(
                f"    vs {bname:14s}: pooled mean   = {m:7.2f} {unit} "
                f"(n={len(peaks):3d})  ->  Ours lower by "
                f"{red_abs:7.2f} {unit} ({red_pct:5.1f}%)"
            )
        red_abs = all_baseline_mean - ours_mean_all
        red_pct = (
            100.0 * red_abs / all_baseline_mean
            if all_baseline_mean > 0 else float("nan")
        )
        print(
            f"    vs ALL baselines pooled: mean = {all_baseline_mean:.2f} {unit} "
            f"(n={len(all_baseline_vals)})  ->  Ours lower by "
            f"{red_abs:.2f} {unit} ({red_pct:.1f}%)"
        )

        pair_means = [
            float(np.mean(peaks))
            for per_task in baseline_peaks_by_task.values()
            for peaks in per_task.values()
            if peaks
        ]
        macro_mean = float(np.mean(pair_means))
        red_abs = macro_mean - ours_mean_all
        red_pct = (
            100.0 * red_abs / macro_mean if macro_mean > 0 else float("nan")
        )
        print(
            f"    vs ALL baselines (macro-avg over task×baseline): "
            f"mean = {macro_mean:.2f} {unit} (n_pairs={len(pair_means)})  ->  "
            f"Ours lower by {red_abs:.2f} {unit} ({red_pct:.1f}%)"
        )


if __name__ == "__main__":
    main()
