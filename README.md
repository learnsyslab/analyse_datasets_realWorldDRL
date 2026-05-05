# analyse_datasets_realWorldDRL

This repository contains utilities for downloading and inspecting LeRobot datasets from Hugging Face.

## Prerequisites

- Linux (project is configured for `linux-64` in `pixi.toml`)
- [Pixi](https://pixi.sh/latest/)

## Environment Setup

From the project root:

```bash
pixi install
```

Run commands inside the Pixi environment with `pixi run ...` (examples below).

## Run `dl_dataset.py`

`dl_dataset.py` downloads a Hugging Face dataset snapshot into the Hugging Face cache directory.

### Default dataset

```bash
pixi run python dl_dataset.py
```

Defaults used by the script:

- repo owner: `OliverHausdoerfer`
- dataset name: `stack_lego_simple_pi05_deploy_2`

### Custom dataset

```bash
pixi run python dl_dataset.py \
  --repo-owner OliverHausdoerfer \
  --dataset-name stack_lego_simple_ditflow_deploy
```

After completion, the script prints the downloaded snapshot path.

## Run `episode_viewer.py`

`episode_viewer.py` opens an OpenCV window and plays the last few seconds of each episode with primary and wrist camera views side by side.

### Start viewer (default dataset)

```bash
pixi run python episode_viewer.py
```

### Start viewer (custom dataset)

Use the same owner and dataset name used for download:

```bash
pixi run python episode_viewer.py \
  --repo-owner OliverHausdoerfer \
  --dataset-name stack_lego_simple_ditflow_deploy
```

The viewer reads data from:

```text
~/.cache/huggingface/hub/datasets--<repo-owner>--<dataset-name>/snapshots/
```

If this path does not exist, run `dl_dataset.py` first.

## Viewer Controls

- `n` or Right Arrow: next episode
- `p` or Left Arrow: previous episode
- `q` or `Esc`: quit
