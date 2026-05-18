#!/usr/bin/env python3
"""
Episode viewer for LeRobot datasets.
Browse episodes and watch the last 5 seconds of videos from both cameras side by side.

Controls:
  - 'n' or RIGHT ARROW: Next episode
  - 'p' or LEFT ARROW: Previous episode
  - 'q' or ESC: Quit
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from huggingface_hub import snapshot_download
from lerobot.datasets import LeRobotDataset
from typing import Optional
import argparse

REPO_ROOT = Path(__file__).parent
DATASETS_DIR = REPO_ROOT / "datasets"


class EpisodeViewer:
    def __init__(self, dataset_path: Path, repo_id: str, display_scale: float = 2.5):
        """Initialize the episode viewer."""
        # Load dataset using LeRobot (root points to the local snapshot dir)
        self.dataset = LeRobotDataset(repo_id, root=str(dataset_path))

        self.display_scale = display_scale

        self.current_episode = 0
        self.total_episodes = self.dataset.num_episodes

        self.repo_id = repo_id
        self.annotations: dict[int, str] = {}
        self.temp_labels_path = REPO_ROOT / "temp_labels.yaml"
        if self.temp_labels_path.exists():
            self.temp_labels_path.unlink()

        # Build episode index mapping
        self._build_episode_ranges()

        print(
            f"Loaded dataset with {self.total_episodes} episodes and {len(self.dataset)} total frames"
        )
        print(f"Annotating: {self.repo_id}")
        print(f"Writing to: {self.temp_labels_path}")
        print("\nLabels (free-form, type as string after each clip):")
        print("  s        success")
        print("  p        partial success")
        print("  f        failure")
        print("  st / pt  add 't' suffix for termination failure")
        print("  s(p)     parenthesised qualifiers are accepted (e.g. p(s?), s(p))")
        print("  n        no annotation marker")
        print("\nPrompt commands:")
        print("  <enter>  replay current clip")
        print("  b        back (previous episode, no save)")
        print("  skip     advance without saving a label")
        print("  q        quit")
        print("  <text>   save <text> as the label and advance")
        print()

    def _build_episode_ranges(self):
        """Build an exact mapping of episode indices to frame ranges."""
        self.episode_ranges = {}
        dataset_len = len(self.dataset)

        for ep_idx in range(self.total_episodes):
            # Find first frame of this episode.
            left, right = 0, dataset_len - 1
            start = -1
            while left <= right:
                mid = (left + right) // 2
                current_ep = int(self.dataset[mid]["episode_index"].item())
                if current_ep == ep_idx:
                    start = mid
                    right = mid - 1
                elif current_ep < ep_idx:
                    left = mid + 1
                else:
                    right = mid - 1

            if start < 0:
                continue

            # Find last frame of this episode.
            left, right = start, dataset_len - 1
            end = start
            while left <= right:
                mid = (left + right) // 2
                current_ep = int(self.dataset[mid]["episode_index"].item())
                if current_ep == ep_idx:
                    end = mid
                    left = mid + 1
                else:
                    right = mid - 1

            self.episode_ranges[ep_idx] = {"start": start, "end": end}

    def get_episode_frames(
        self, episode_idx: int, last_n_seconds: float = 3.0, fps: float = 30.0
    ) -> tuple:
        """Get the last n seconds of frames for an episode."""
        if episode_idx not in self.episode_ranges:
            return [], [], 0

        start_idx = self.episode_ranges[episode_idx]["start"]
        end_idx = self.episode_ranges[episode_idx]["end"]
        total_frames = end_idx - start_idx + 1

        # Calculate how many frames we need
        num_frames_to_get = int(last_n_seconds * fps)
        num_frames_to_get = min(num_frames_to_get, total_frames)

        # Get the start index for the last N frames
        frame_start_idx = end_idx - num_frames_to_get + 1

        frames_primary = []
        frames_wrist = []

        for frame_idx in range(frame_start_idx, end_idx + 1):
            try:
                frame_data = self.dataset[frame_idx]

                # Extract both camera images
                img_primary = frame_data.get("observation.images.primary")
                if img_primary is not None and hasattr(img_primary, "cpu"):
                    img_primary = img_primary.cpu().numpy()
                    if len(img_primary.shape) == 3 and img_primary.shape[0] in [1, 3]:
                        # CHW format - convert to HWC
                        img_primary = np.moveaxis(img_primary, 0, -1)
                    frames_primary.append(img_primary)

                img_wrist = frame_data.get("observation.images.wrist")
                if img_wrist is not None and hasattr(img_wrist, "cpu"):
                    img_wrist = img_wrist.cpu().numpy()
                    if len(img_wrist.shape) == 3 and img_wrist.shape[0] in [1, 3]:
                        # CHW format - convert to HWC
                        img_wrist = np.moveaxis(img_wrist, 0, -1)
                    frames_wrist.append(img_wrist)
            except Exception as e:
                print(f"Error loading frame {frame_idx}: {e}")
                continue

        return frames_primary, frames_wrist, total_frames

    def display_episode(self, episode_idx: int):
        """Display the clip in a replay loop until the user labels/skips/quits."""
        print(
            f"Loading episode {episode_idx + 1}/{self.total_episodes}...",
            end="",
            flush=True,
        )

        try:
            frames_primary, frames_wrist, total_frames = self.get_episode_frames(
                episode_idx
            )

            if not frames_primary or not frames_wrist:
                print(f" ERROR: No frames loaded")
                return None

            print(f" OK ({len(frames_primary)} frames, {total_frames} total)")

            displayed_frames = min(len(frames_primary), len(frames_wrist))
            clip_start_frame = total_frames - displayed_frames + 1

            while True:
                play_result = self._play_frames_side_by_side(
                    frames_primary,
                    frames_wrist,
                    episode_idx,
                    total_episode_frames=total_frames,
                    clip_start_frame=clip_start_frame,
                )
                if play_result == "quit":
                    return "quit"

                action = self._prompt_for_label(episode_idx)
                if action == "replay":
                    continue
                if action in ("prev", "skip", "quit"):
                    return action
                if action.startswith("label:"):
                    label = action[len("label:") :]
                    self._save_label(episode_idx, label)
                    return "next"
                return action
        except Exception as e:
            print(f" ERROR: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _prompt_for_label(self, episode_idx: int) -> str:
        """Read a label / navigation command from the terminal."""
        prompt = (
            f"Episode {episode_idx + 1}/{self.total_episodes} "
            f"— label / b / skip / q (enter=replay): "
        )
        try:
            raw = input(prompt)
        except EOFError:
            return "quit"
        text = raw.strip()
        if text == "":
            return "replay"
        if text == "q":
            return "quit"
        if text == "b":
            return "prev"
        if text == "skip":
            return "skip"
        return f"label:{text}"

    def _save_label(self, episode_idx: int, label: str) -> None:
        """Persist annotations to temp_labels.yaml atomically."""
        self.annotations[episode_idx] = label
        data = {self.repo_id: dict(sorted(self.annotations.items()))}
        tmp = self.temp_labels_path.with_suffix(".yaml.tmp")
        tmp.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=False))
        tmp.replace(self.temp_labels_path)
        print(f"  saved: ep {episode_idx} -> {label!r}")

    def _play_frames_side_by_side(
        self,
        frames_primary: list,
        frames_wrist: list,
        episode_idx: int,
        fps: float = 30.0,
        total_episode_frames: Optional[int] = None,
        clip_start_frame: int = 1,
    ):
        """Display frames side by side with opencv."""
        window_name = "Episode Viewer"

        # Create a resizable window once. WINDOW_NORMAL lets us call
        # resizeWindow() — WINDOW_AUTOSIZE would ignore that.
        if not getattr(self, "_window_created", False):
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            self._window_created = True

        for frame_idx, (frame_p, frame_w) in enumerate(
            zip(frames_primary, frames_wrist)
        ):
            # Ensure frames are in proper format
            if frame_p.dtype == np.float32 or frame_p.dtype == np.float64:
                frame_p = (frame_p * 255).astype(np.uint8)
            if frame_w.dtype == np.float32 or frame_w.dtype == np.float64:
                frame_w = (frame_w * 255).astype(np.uint8)

            # Handle single channel images
            if len(frame_p.shape) == 2:
                frame_p = cv2.cvtColor(frame_p, cv2.COLOR_GRAY2BGR)
            elif frame_p.shape[2] == 3:
                # Check if it's RGB and convert to BGR
                frame_p = cv2.cvtColor(frame_p, cv2.COLOR_RGB2BGR)

            if len(frame_w.shape) == 2:
                frame_w = cv2.cvtColor(frame_w, cv2.COLOR_GRAY2BGR)
            elif frame_w.shape[2] == 3:
                frame_w = cv2.cvtColor(frame_w, cv2.COLOR_RGB2BGR)

            # Resize wrist to match primary height for better side-by-side display
            target_height = frame_p.shape[0]
            target_width = int(frame_w.shape[1] * target_height / frame_w.shape[0])
            frame_w_resized = cv2.resize(frame_w, (target_width, target_height))

            # Create side-by-side display
            side_by_side = np.hstack([frame_p, frame_w_resized])

            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                side_by_side, "Primary Camera", (10, 30), font, 1, (0, 255, 0), 2
            )
            cv2.putText(
                side_by_side,
                "Wrist Camera",
                (frame_p.shape[1] + 10, 30),
                font,
                1,
                (0, 255, 0),
                2,
            )

            # Show episode info
            episode_info = f"Episode {episode_idx + 1}/{self.total_episodes}"
            cv2.putText(
                side_by_side,
                episode_info,
                (10, side_by_side.shape[0] - 40),
                font,
                0.7,
                (100, 200, 100),
                1,
            )

            # Show window info
            cv2.putText(
                side_by_side,
                "Type label in terminal after clip  (q/ESC: quit)",
                (10, side_by_side.shape[0] - 10),
                font,
                0.6,
                (200, 200, 200),
                1,
            )

            # Show current frame position within displayed clip
            total_original = (
                total_episode_frames
                if total_episode_frames is not None
                else len(frames_primary)
            )
            current_original_frame = clip_start_frame + frame_idx
            frame_info = f"Frame {current_original_frame}/{total_original}"
            cv2.putText(
                side_by_side,
                frame_info,
                (10, side_by_side.shape[0] - 70),
                font,
                0.7,
                (100, 200, 100),
                1,
            )

            # Upscale the composite for a larger window
            if self.display_scale != 1.0:
                side_by_side = cv2.resize(
                    side_by_side,
                    None,
                    fx=self.display_scale,
                    fy=self.display_scale,
                    interpolation=cv2.INTER_LINEAR,
                )

            # Show frame
            cv2.imshow(window_name, side_by_side)

            # Force the window size to match the (upscaled) image. Must happen
            # after imshow on some OpenCV builds. Done once per playback session.
            if not getattr(self, "_window_sized", False):
                h, w = side_by_side.shape[:2]
                cv2.resizeWindow(window_name, w, h)
                self._window_sized = True
                print(
                    f"  display: native {frame_p.shape[1] // 1}x{frame_p.shape[0]} (primary) "
                    f"+ {frame_w.shape[1]}x{frame_w.shape[0]} (wrist) "
                    f"-> window {w}x{h} @ scale {self.display_scale}"
                )

            # Wait for the appropriate frame duration (in ms)
            frame_duration_ms = int(1000 / fps)
            key = cv2.waitKey(frame_duration_ms) & 0xFF

            if key == ord("q") or key == 27:  # q or ESC
                return "quit"

        return "clip_ended"

    def run(self):
        """Main loop for browsing episodes."""
        try:
            while True:
                action = self.display_episode(self.current_episode)

                if action == "quit":
                    print("\nExiting...")
                    break
                elif action in ("next", "skip"):
                    if self.current_episode + 1 >= self.total_episodes:
                        print("\nReached end of dataset. Exiting...")
                        break
                    self.current_episode += 1
                elif action == "prev":
                    self.current_episode = max(0, self.current_episode - 1)
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Episode viewer for LeRobot datasets")
    parser.add_argument(
        "--repo-owner",
        default="OliverHausdoerfer",
        help="Hugging Face dataset repository owner",
    )
    parser.add_argument(
        "--dataset-name",
        default="ditflow_siemens_difficult_generalization_deploy",
        help="Dataset name",
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=2.5,
        help="Upscale factor applied to the side-by-side composite (1.0 = original)",
    )
    args = parser.parse_args()

    repo_id = f"{args.repo_owner}/{args.dataset_name}"
    dataset_path = DATASETS_DIR / repo_id

    if dataset_path.exists():
        print(f"Found dataset at {dataset_path}")
    else:
        print(f"Dataset not found locally; downloading {repo_id} to {dataset_path}")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(dataset_path),
        )

    print(f"Using dataset path: {dataset_path}")

    viewer = EpisodeViewer(dataset_path, repo_id, display_scale=args.display_scale)
    viewer.run()
