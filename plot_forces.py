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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle

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
SHELF_OURS_DIR = DATASETS_DIR / "stats_ours_shelf_task"
FORCE_OUTPUT_PATH = REPO_ROOT / "forces_boxplot.pdf"
TORQUE_OUTPUT_PATH = REPO_ROOT / "torques_boxplot.pdf"
SINGLE_COLUMN_OUTPUT_PATH = REPO_ROOT / "forces_boxplot_single_column.pdf"
COMBINED_OUTPUT_PATH = REPO_ROOT / "forces_torques_boxplot.pdf"

# Figures are drawn at 5x scale (see font.size above): a 40 in wide figure maps
# to the paper's \textwidth (516 pt), so a single column (252 pt) is ~19.5 in.
TEXT_WIDTH_IN = 40
COLUMN_WIDTH_IN = TEXT_WIDTH_IN * 252 / 516

FT_COL = "observation.state.sensors_bota_ft_sensor"

# FT sensor is 6-dim: [Fx, Fy, Fz, Tx, Ty, Tz].
METRICS = {
    "force": {
        "slice": slice(0, 3),
        "json_key": "force_N",
        "ylabel": "Episodic max.\nForces [N]",
        "output": FORCE_OUTPUT_PATH,
    },
    "torque": {
        "slice": slice(3, 6),
        "json_key": "torque_Nm",
        "ylabel": "Episodic max.\nTorques [Nm]",
        "output": TORQUE_OUTPUT_PATH,
    },
}

POLICY_LABELS = {
    "Pi05": "Finetuned Pi05",
    "Ours (dataset)": "Ours\n(learned policy)",
    "Ditflow Novice": "DiTFlow (novice)",
    "Diffusion": "Diffusion",
    "Ditflow": "DiTFlow",
}
OURS_JSON_LABEL = "Ours\n(push down)"
# Single-line labels for narrow layouts, where the tick labels are rotated.
POLICY_LABELS_ROTATED = {
    **POLICY_LABELS,
    "Ours (dataset)": "Ours (learned policy)",
}
OURS_JSON_LABEL_ROTATED = "Ours (push down)"

# Baseline policy label -> task instance. Order here defines the x-axis order;
# "Ours" is appended last from its eval results JSON (see OURS_JSON below).
OURS_TASK_SIEMENS = config.OursSiemensGeneralization1()
POLICIES_SIEMENS = {
    "Diffusion": config.SiemensDiffusion(),
    "Ditflow": config.SiemensDitflow(),
    # "Ditflow Novice": config.LegoSimpleDitflowNovice(),
    "Pi05": config.SiemensPi05(),
    "Ours (dataset)": OURS_TASK_SIEMENS,
}

OURS_TASK_LEGO = config.OursLegoSimple()
OURS_TASK_LEGO_JSON = config.OursLegoFt20()
POLICIES_LEGO = {
    "Diffusion": config.LegoSimpleDiffusion(),
    "Ditflow": config.LegoSimpleDitflow(),
    "Ditflow Novice": config.LegoSimpleDitflowNovice(),
    "Pi05": config.LegoSimplePi05(),
    "Ours (dataset)": OURS_TASK_LEGO,
}

POLICIES_SHELF = {
    "Diffusion": config.ShelfDiffusion(),
    "Ditflow": config.ShelfDitflow(),
    "Ditflow Novice": config.ShelfDtiflowJim(),
    "Pi05": config.ShelfPi05(),
}

OURS_JSON_KEY = "__ours_json__"

CANONICAL_COLUMNS = (
    "Diffusion",
    "Ditflow",
    "Ditflow Novice",
    "Pi05",
    "Ours (dataset)",
    OURS_JSON_KEY,
)

TASKS = {
    "Fan cover": {
        "gradient": ("#523861", "#874f6e"),  # bottom -> top
        "policies": POLICIES_SIEMENS,
        "ours_json_task": OURS_TASK_SIEMENS,
    },
    "Lego": {
        "gradient": ("#c36c43", "#d78d57"),  # bottom -> top
        "policies": POLICIES_LEGO,
        "ours_json_task": OURS_TASK_LEGO_JSON,
    },
    "Shelf stocking": {
        "gradient": ("#b73779", "#de4968"),  # magma mid-band (pink -> red)
        "policies": POLICIES_SHELF,
        "ours_json_task": None,
        "ours_pushdown_json": SHELF_OURS_DIR / "ft_statistics.json",
        "ours_learned_json": SHELF_OURS_DIR / "ft_statistics_policy.json",
    },
}


_PEAK_CACHE: dict[tuple[str, int, int], list[float]] = {}


def episode_peak_forces(repo_id: str, axes: slice = slice(0, 3)) -> list[float]:
    """Download a dataset and return the per-episode peak magnitude of the
    FT-sensor sub-vector selected by ``axes`` (default: force = first 3 axes)."""
    cache_key = (repo_id, axes.start, axes.stop)
    if cache_key in _PEAK_CACHE:
        return _PEAK_CACHE[cache_key]

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

    _PEAK_CACHE[cache_key] = list(peak.values())
    return _PEAK_CACHE[cache_key]


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


class _GradientHandle:
    """Legend handle marker carrying a (bottom, top) color pair."""
    def __init__(self, gradient: tuple[str, str], label: str):
        self.gradient = gradient
        self.label = label

    def get_label(self) -> str:
        return self.label


class _GradientHandler(HandlerBase):
    """Render a `_GradientHandle` as a vertical gradient swatch."""
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        cmap = LinearSegmentedColormap.from_list(
            "legend_grad", list(orig_handle.gradient)
        )
        n = 64
        x0, y0 = -xdescent, -ydescent
        strip_h = height / n
        artists = []
        for i in range(n):
            color = cmap(i / max(1, n - 1))
            artists.append(
                Rectangle(
                    (x0, y0 + i * strip_h),
                    width,
                    strip_h + 0.5,  # overlap to avoid hairline gaps
                    facecolor=color,
                    edgecolor="none",
                    transform=trans,
                )
            )
        artists.append(
            Rectangle(
                (x0, y0),
                width,
                height,
                facecolor="none",
                edgecolor="black",
                linewidth=0.5,
                transform=trans,
            )
        )
        return artists


def _fill_box_with_gradient(
    ax, box_patch, gradient: tuple[str, str], horizontal: bool = False
) -> None:
    """Replace a solid box facecolor with a gradient along the value axis
    (bottom -> top, or left -> right for horizontal boxes)."""
    color_bottom, color_top = gradient
    cmap = LinearSegmentedColormap.from_list("box_grad", [color_bottom, color_top])
    grad = np.linspace(0, 1, 256)
    grad = grad.reshape(1, -1) if horizontal else grad.reshape(-1, 1)
    verts = box_patch.get_path().vertices
    x0, x1 = float(verts[:, 0].min()), float(verts[:, 0].max())
    y0, y1 = float(verts[:, 1].min()), float(verts[:, 1].max())
    box_patch.set_facecolor("none")
    im = ax.imshow(
        grad,
        aspect="auto",
        cmap=cmap,
        extent=(x0, x1, y0, y1),
        origin="lower",
        zorder=box_patch.get_zorder() - 0.1,
    )
    im.set_clip_path(box_patch)


def collect_peaks(column_key: str, task_entry: dict, cfg: dict) -> list[float] | None:
    """Return per-episode peaks for one (column, task) cell, or None if absent."""
    if column_key == OURS_JSON_KEY:
        explicit = task_entry.get("ours_pushdown_json")
        if explicit is not None:
            return episode_peaks_from_json(explicit, cfg["json_key"]) if Path(explicit).is_file() else None
        t = task_entry["ours_json_task"]
        if t is None:
            return None
        json_path = DATASETS_DIR / t.datasets[0] / "eval_results.json"
        if not json_path.is_file():
            return None
        return episode_peaks_from_json(json_path, cfg["json_key"])
    if column_key == "Ours (dataset)":
        explicit = task_entry.get("ours_learned_json")
        if explicit is not None:
            return episode_peaks_from_json(explicit, cfg["json_key"]) if Path(explicit).is_file() else None
    task = task_entry["policies"].get(column_key)
    if task is None:
        return None
    peaks: list[float] = []
    for repo_id in task.datasets:
        peaks.extend(episode_peak_forces(repo_id, axes=cfg["slice"]))
    return peaks


def collect_cells(metric_name: str):
    """Collect per-episode peaks for every (column, task) cell of one metric.

    Returns ``(cells, missing)`` with ``cells`` = [(column_idx, task_idx, task, vals)]
    and ``missing`` = [(column_idx, task_idx)] for cells without data.
    """
    cfg = METRICS[metric_name]
    cells: list[tuple[int, int, str, list[float]]] = []
    missing: list[tuple[int, int]] = []
    for c, col in enumerate(CANONICAL_COLUMNS):
        col_label = OURS_JSON_LABEL if col == OURS_JSON_KEY else col
        for i, tname in enumerate(TASKS):
            print(f"  [{col_label}] {tname}...", flush=True)
            vals = collect_peaks(col, TASKS[tname], cfg)
            if vals is None or len(vals) == 0:
                missing.append((c, i))
                continue
            cells.append((c, i, tname, vals))
    return cells, missing


def _format_label(label: str) -> str:
    if label.startswith("Ours"):
        return r"$\mathbf{Ours}$" + label[len("Ours"):]
    return label


def draw_boxes(
    ax, cells, missing, horizontal: bool = False, labels=POLICY_LABELS,
    ours_json_label: str = OURS_JSON_LABEL, rotation: float = 0.0,
) -> None:
    """Draw grouped gradient boxplots (one group per policy, one box per task).

    With ``horizontal=True`` policies run top -> bottom on the y-axis and the
    metric is on the x-axis. ``rotation`` tilts the policy tick labels
    (vertical layout only).
    """
    n_tasks = len(TASKS)
    group_width = 0.8
    box_width = group_width / n_tasks

    def _pos(c: int, i: int) -> float:
        return c + (i - (n_tasks - 1) / 2) * box_width

    for c, i, tname, vals in cells:
        bp = ax.boxplot(
            [vals],
            positions=[_pos(c, i)],
            widths=box_width * 0.72,
            patch_artist=True,
            showmeans=True,
            medianprops=dict(color="black"),
            orientation="horizontal" if horizontal else "vertical",
        )
        _fill_box_with_gradient(ax, bp["boxes"][0], TASKS[tname]["gradient"], horizontal)

    separator = ax.axhline if horizontal else ax.axvline
    for x in range(1, len(CANONICAL_COLUMNS)):
        separator(x - 0.5, color="grey", linewidth=0.8, alpha=0.4, zorder=0)

    # Set value limits ourselves (imshow gradient fills clobber autoscale).
    all_vals = [v for _, _, _, vals in cells for v in vals]
    if all_vals:
        vmin, vmax = min(all_vals), max(all_vals)
        margin = (vmax - vmin) * 0.05
        set_vlim = ax.set_xlim if horizontal else ax.set_ylim
        set_vlim(max(0.0, vmin - margin), vmax + margin)

    for c, i in missing:
        if horizontal:
            ax.text(0.01, _pos(c, i), "N/A", transform=ax.get_yaxis_transform(),
                    ha="left", va="center", color="grey",
                    fontsize=plt.rcParams["font.size"] * 0.7)
        else:
            ax.text(_pos(c, i), 0.01, "N/A", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", color="grey",
                    fontsize=plt.rcParams["font.size"] * 0.7)

    tick_labels = [
        _format_label(ours_json_label if col == OURS_JSON_KEY else labels.get(col, col))
        for col in CANONICAL_COLUMNS
    ]
    ticks = range(len(CANONICAL_COLUMNS))
    if horizontal:
        ax.set_yticks(ticks)
        ax.set_yticklabels(tick_labels)
        ax.set_ylim(len(CANONICAL_COLUMNS) - 0.5, -0.5)  # first policy on top
        ax.grid(axis="x", alpha=0.3)
    else:
        ax.set_xticks(ticks)
        if rotation:
            ax.set_xticklabels(tick_labels, rotation=rotation, ha="right", rotation_mode="anchor")
        else:
            ax.set_xticklabels(tick_labels)
        ax.set_xlim(-0.5, len(CANONICAL_COLUMNS) - 0.5)
        ax.grid(axis="y", alpha=0.3)


def _task_legend(target, **kwargs):
    handles = [_GradientHandle(TASKS[t]["gradient"], t) for t in TASKS]
    return target.legend(
        handles=handles,
        handler_map={_GradientHandle: _GradientHandler()},
        ncol=len(handles),
        **kwargs,
    )


def plot_metric(metric_name: str) -> None:
    cfg = METRICS[metric_name]
    cells, missing = collect_cells(metric_name)

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 8))
    draw_boxes(ax, cells, missing)
    ax.set_ylabel(cfg["ylabel"])
    _task_legend(ax, loc="best")
    fig.tight_layout()
    fig.savefig(cfg["output"])
    plt.close(fig)
    print(f"Wrote plot to {cfg['output']}")


def _draw_rotated(ax, metric_name: str) -> None:
    """Vertical boxes with rotated single-line policy labels (narrow layouts)."""
    cells, missing = collect_cells(metric_name)
    draw_boxes(ax, cells, missing, labels=POLICY_LABELS_ROTATED,
               ours_json_label=OURS_JSON_LABEL_ROTATED, rotation=30)
    ax.set_ylabel(METRICS[metric_name]["ylabel"])


def plot_single_column() -> None:
    """Forces only, vertical boxes, sized for one paper column."""
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 15))
    _draw_rotated(ax, "force")
    _task_legend(ax, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                 frameon=False, handlelength=1.2, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(SINGLE_COLUMN_OUTPUT_PATH)
    plt.close(fig)
    print(f"Wrote plot to {SINGLE_COLUMN_OUTPUT_PATH}")


def plot_combined() -> None:
    """Forces (left) and torques (right) side by side, sized for the full text width."""
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 11))
    for ax, metric_name in zip(axes, ("force", "torque")):
        _draw_rotated(ax, metric_name)
    _task_legend(fig, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(COMBINED_OUTPUT_PATH)
    plt.close(fig)
    print(f"Wrote plot to {COMBINED_OUTPUT_PATH}")


def main() -> None:
    for metric_name in METRICS:
        print(f"=== {metric_name} ===")
        plot_metric(metric_name)
    print("=== single column ===")
    plot_single_column()
    print("=== combined ===")
    plot_combined()


if __name__ == "__main__":
    main()
