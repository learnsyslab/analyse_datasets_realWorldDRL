"""Export the per-episode state values V(s)=Q(s, pi(s)) behind
plot_state_values_lerobot.py as raw arrays (no plotting), so the figure can be
re-plotted elsewhere. Reuses that script's loading/evaluation code unchanged.

Output: <output_dir>/state_values_raw.npz with
  success_<i>, truncated_<i>: 1-D float32 arrays of V(s_t) per episode
  success_episode_ids, truncated_episode_ids: episode indices (from file names)
  meta: JSON string (dataset, checkpoint, counts)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import plot_state_values_lerobot as psv  # noqa: E402


def main() -> None:
    args = psv.parse_args()
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path("plots") / f"state_values_{args.dataset_dir.name}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    episode_files = psv.find_episode_files(args.dataset_dir)
    shared_encoder, actor, q_functions, config = psv.load_models(args.checkpoint_dir, device)
    expected_obs_dim = (
        config.actor_nonvision_input_dim + config.vision_head_input_dim * config.n_cameras
    )

    out: dict[str, np.ndarray] = {}
    ids = {"success": [], "truncated": []}
    other = 0
    for episode_path in episode_files:
        observations, rewards = psv.load_episode_observations_and_rewards(episode_path)
        assert observations.shape[1] == expected_obs_dim, episode_path
        finite = rewards[np.isfinite(rewards)]
        if finite.size == 0:
            continue
        kind = psv.classify_episode(float(finite[-1]))
        if kind == "success":
            if len(observations) <= 1:
                continue
            observations = observations[:-1]
        elif kind != "truncated":
            other += 1
            continue
        values = psv.compute_state_values(
            observations=observations, shared_encoder=shared_encoder, actor=actor,
            q_functions=q_functions, device=device,
        )
        ep_id = psv._extract_episode_id(episode_path)
        out[f"{kind}_{len(ids[kind])}"] = values.astype(np.float32)
        ids[kind].append(ep_id)

    meta = {
        "dataset_dir": str(args.dataset_dir), "checkpoint_dir": str(args.checkpoint_dir),
        "n_success": len(ids["success"]), "n_truncated": len(ids["truncated"]), "n_other": other,
        "n_q_functions": len(q_functions),
        "note": "values = mean over critic ensemble of Q(s_t, pi(s_t)); successful episodes exclude the terminal observation",
    }
    out["success_episode_ids"] = np.asarray(ids["success"], dtype=np.int64)
    out["truncated_episode_ids"] = np.asarray(ids["truncated"], dtype=np.int64)
    out["meta"] = np.asarray(json.dumps(meta))
    path = output_dir / "state_values_raw.npz"
    np.savez_compressed(path, **out)
    print(f"Saved {path}: {meta}")


if __name__ == "__main__":
    main()
