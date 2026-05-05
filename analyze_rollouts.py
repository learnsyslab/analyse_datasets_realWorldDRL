"""Compute average episode length for successful rollouts per task.

Successful = annotation in labels.yaml is exactly "s".
Datasets are downloaded in full (incl. videos) into ./datasets/<repo_id>/.
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import dataclasses
import inspect
import json
import logging
from pathlib import Path
from statistics import mean, median

import numpy as np
import pandas as pd
import yaml
from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars

import config

disable_progress_bars()
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

REPO_ROOT = Path(__file__).parent
DATASETS_DIR = REPO_ROOT / "datasets"
LABELS_PATH = REPO_ROOT / "labels.yaml"
OUTPUT_PATH = REPO_ROOT / "analyze_rollouts_output.txt"

TASKS = [
    cls()
    for _, cls in inspect.getmembers(config, inspect.isclass)
    if dataclasses.is_dataclass(cls) and cls.__module__ == config.__name__
]


FT_COL = "observation.state.sensors_bota_ft_sensor"


def load_dataset(
    repo_id: str,
) -> tuple[pd.Series, float, dict[int, float], dict[int, float]]:
    """Return (episode_lengths_by_index, fps, peak_force_by_ep, peak_torque_by_ep).

    peak_*_by_ep are populated for every episode that appears in the data parquets.
    """
    local_dir = DATASETS_DIR / repo_id
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
    )
    meta_files = sorted((local_dir / "meta" / "episodes").rglob("*.parquet"))
    if not meta_files:
        raise FileNotFoundError(f"No episode parquet files found under {local_dir}")
    meta_df = pd.concat([pd.read_parquet(p) for p in meta_files], ignore_index=True)
    lengths = meta_df.set_index("episode_index")["length"]

    with open(local_dir / "meta" / "info.json") as f:
        info = json.load(f)
    fps = float(info["fps"])

    peak_force: dict[int, float] = {}
    peak_torque: dict[int, float] = {}
    data_files = sorted((local_dir / "data").rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No data parquet files found under {local_dir}")
    for p in data_files:
        df = pd.read_parquet(p, columns=["episode_index", FT_COL])
        if df.empty:
            continue
        vecs = np.stack(list(df[FT_COL].to_numpy()))
        force_mag = np.linalg.norm(vecs[:, :3], axis=1)
        torque_mag = np.linalg.norm(vecs[:, 3:], axis=1)
        ep_idx = df["episode_index"].to_numpy()
        for ep in np.unique(ep_idx):
            mask = ep_idx == ep
            f_max = float(force_mag[mask].max())
            t_max = float(torque_mag[mask].max())
            ep_int = int(ep)
            peak_force[ep_int] = max(peak_force.get(ep_int, f_max), f_max)
            peak_torque[ep_int] = max(peak_torque.get(ep_int, t_max), t_max)

    return lengths, fps, peak_force, peak_torque


def classify(label: str) -> str:
    """Map a raw annotation to 'success', 'partial', or 'failure'.

    Rules: leading char is primary; `st`/`pt` demote to partial; `n` is partial.
    """
    if label in ("st", "pt"):
        return "partial"
    if label == "n":
        return "partial"
    head = label[0]
    if head == "s":
        return "success"
    if head == "p":
        return "partial"
    if head == "f":
        return "failure"
    raise ValueError(f"Unknown label: {label!r}")


def stats(values):
    if not values:
        return None
    return {
        "min": min(values),
        "median": median(values),
        "mean": mean(values),
        "max": max(values),
    }


def fmt_stats(s, prec):
    if s is None:
        return "n/a"
    return f"min={s['min']:.{prec}f} median={s['median']:.{prec}f} mean={s['mean']:.{prec}f} max={s['max']:.{prec}f}"


def main() -> None:
    with open(LABELS_PATH) as f:
        labels = yaml.safe_load(f)

    print("Downloading / verifying datasets...", flush=True)

    results: list[dict] = []

    for task in TASKS:
        task_name = type(task).__name__
        print(f"  {task_name}...", flush=True)
        succ_frames: list[int] = []
        succ_seconds: list[float] = []
        succ_peak_force: list[float] = []
        succ_peak_torque: list[float] = []
        all_peak_force: list[float] = []
        all_peak_torque: list[float] = []
        n_labeled = 0
        n_partial = 0
        n_failure = 0

        for repo_id in getattr(task, "datasets"):
            ds_labels = labels.get(repo_id, {})
            ep_class = {ep: classify(lbl) for ep, lbl in ds_labels.items()}
            success_eps = {ep for ep, c in ep_class.items() if c == "success"}
            partial_eps = {ep for ep, c in ep_class.items() if c == "partial"}
            failure_eps = {ep for ep, c in ep_class.items() if c == "failure"}
            n_labeled += len(ep_class)
            n_partial += len(partial_eps)
            n_failure += len(failure_eps)
            lengths_by_idx, fps, peak_force, peak_torque = load_dataset(repo_id)

            succ_frames.extend(int(lengths_by_idx.loc[ep]) for ep in success_eps)
            succ_seconds.extend(int(lengths_by_idx.loc[ep]) / fps for ep in success_eps)
            succ_peak_force.extend(peak_force[ep] for ep in success_eps if ep in peak_force)
            succ_peak_torque.extend(peak_torque[ep] for ep in success_eps if ep in peak_torque)
            all_peak_force.extend(peak_force.values())
            all_peak_torque.extend(peak_torque.values())

        n_success = len(succ_frames)
        results.append(
            {
                "name": task_name,
                "n_success": n_success,
                "n_partial": n_partial,
                "n_failure": n_failure,
                "n_labeled": n_labeled,
                "pct_success": (n_success / n_labeled * 100) if n_labeled else None,
                "pct_partial": ((n_success + n_partial) / n_labeled * 100) if n_labeled else None,
                "frames": stats(succ_frames),
                "seconds": stats(succ_seconds),
                "succ_force": stats(succ_peak_force),
                "succ_torque": stats(succ_peak_torque),
                "all_force": stats(all_peak_force),
                "all_torque": stats(all_peak_torque),
            }
        )

    report_lines: list[str] = []
    def emit(line: str = "") -> None:
        print(line)
        report_lines.append(line)

    emit()
    emit("=" * 100)
    emit(" Per-task stats (successful rollouts have label == 's')")
    emit("=" * 100)
    for r in results:
        pct_s = f"{r['pct_success']:.1f}%" if r["pct_success"] is not None else "n/a"
        pct_p = f"{r['pct_partial']:.1f}%" if r["pct_partial"] is not None else "n/a"
        emit(
            f"{r['name']}  (n_success={r['n_success']}, n_partial={r['n_partial']}, "
            f"n_failure={r['n_failure']}, n_labeled={r['n_labeled']}, "
            f"success={pct_s}, partial={pct_p})"
        )
        emit(f"    frames                : {fmt_stats(r['frames'], 1)}")
        emit(f"    seconds               : {fmt_stats(r['seconds'], 2)}")
        emit(f"    peak |F| success (N)  : {fmt_stats(r['succ_force'], 2)}")
        emit(f"    peak |M| success (N·m): {fmt_stats(r['succ_torque'], 3)}")
        emit(f"    peak |F| all     (N)  : {fmt_stats(r['all_force'], 2)}")
        emit(f"    peak |M| all     (N·m): {fmt_stats(r['all_torque'], 3)}")
        emit()

    OUTPUT_PATH.write_text("\n".join(report_lines) + "\n")
    print(f"Wrote report to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
