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
from pathlib import Path
from lerobot.datasets import LeRobotDataset
from typing import Optional
import argparse
import sys


class EpisodeViewer:
    def __init__(self, dataset_path: Path):
        """Initialize the episode viewer."""
        # Load dataset using LeRobot
        self.dataset = LeRobotDataset(str(dataset_path))

        self.current_episode = 0
        self.total_episodes = self.dataset.num_episodes

        # Build episode index mapping
        self._build_episode_ranges()

        print(
            f"Loaded dataset with {self.total_episodes} episodes and {len(self.dataset)} total frames"
        )
        print("\nControls:")
        print("  n / RIGHT ARROW - Next episode")
        print("  p / LEFT ARROW - Previous episode")
        print("  q / ESC - Quit")
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
        """Display the last 3 seconds of an episode with side-by-side cameras."""
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

            # Play frames
            displayed_frames = min(len(frames_primary), len(frames_wrist))
            clip_start_frame = total_frames - displayed_frames + 1
            action = self._play_frames_side_by_side(
                frames_primary,
                frames_wrist,
                episode_idx,
                total_episode_frames=total_frames,
                clip_start_frame=clip_start_frame,
            )

            return action
        except Exception as e:
            print(f" ERROR: {e}")
            import traceback

            traceback.print_exc()
            return None

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

        # Create window once if it doesn't exist
        try:
            cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
        except:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

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
                "n/RIGHT: Next  p/LEFT: Prev  q/ESC: Quit",
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

            # Show frame
            cv2.imshow(window_name, side_by_side)

            # Wait for the appropriate frame duration (in ms)
            frame_duration_ms = int(1000 / fps)
            key = cv2.waitKey(frame_duration_ms) & 0xFF

            if key == ord("q") or key == 27:  # q or ESC
                return "quit"
            elif key == ord("n") or key == 83:  # n or RIGHT Arrow
                return "next"
            elif key == ord("p") or key == 81:  # p or LEFT Arrow
                return "prev"

        return None

    def run(self):
        """Main loop for browsing episodes."""
        try:
            while True:
                action = self.display_episode(self.current_episode)

                if action == "quit":
                    print("\nExiting...")
                    break
                elif action == "next":
                    self.current_episode = (
                        self.current_episode + 1
                    ) % self.total_episodes
                elif action == "prev":
                    self.current_episode = (
                        self.current_episode - 1
                    ) % self.total_episodes
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
        default="stack_lego_simple_pi05_deploy_2",
        help="Dataset name",
    )
    args = parser.parse_args()

    snapshots_dir = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"datasets--{args.repo_owner}--{args.dataset_name}"
        / "snapshots"
    )

    if not snapshots_dir.exists() or not snapshots_dir.is_dir():
        print(f"Error: Snapshots directory not found at {snapshots_dir}")
        sys.exit(1)

    snapshot_subdirs = sorted([p for p in snapshots_dir.iterdir() if p.is_dir()])
    if not snapshot_subdirs:
        print(f"Error: No snapshot subdirectories found in {snapshots_dir}")
        sys.exit(1)

    dataset_path = snapshot_subdirs[0]

    print(f"Using dataset path: {dataset_path}")

    viewer = EpisodeViewer(dataset_path)
    viewer.run()
