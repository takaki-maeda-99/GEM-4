"""Taco Play dataset inspection + sample visualization.

Prints dataset stats and saves scene/wrist frames as PNG + animated GIF.
Output: runs/micro_smoke_logs/samples/

Run:
  .venv-gemma4/bin/python scripts/gemma4/viz_taco_sample.py
"""
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
import tensorflow_datasets as tfds
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = REPO_ROOT / "data" / "stage3_openx"
OUT_DIR = REPO_ROOT / "runs" / "micro_smoke_logs" / "samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=== Taco Play dataset info ===")
    builder = tfds.builder("taco_play", data_dir=str(DATA_ROOT))
    info = builder.info
    print(f"num episodes (train): {info.splits['train'].num_examples}")
    print(f"features:")
    for k in info.features["steps"].feature.keys():
        print(f"  {k}")

    # Load a few episodes and inspect
    ds = tfds.load("taco_play", data_dir=str(DATA_ROOT), split="train", shuffle_files=False)

    print("\n=== First 3 episodes: length + language ===")
    step_counts = []
    for ep_idx, episode in enumerate(ds.take(3)):
        steps_list = list(episode["steps"].as_numpy_iterator())
        step_counts.append(len(steps_list))
        lang = ""
        for step in steps_list:
            lang_key = "language_instruction" if "language_instruction" in step else (
                "natural_language_instruction" if "natural_language_instruction" in step else None
            )
            if lang_key and step.get(lang_key) is not None:
                raw = step[lang_key]
                if isinstance(raw, bytes):
                    lang = raw.decode("utf-8", errors="replace").strip()
                else:
                    lang = str(raw).strip()
                break
        print(f"  ep{ep_idx}: {len(steps_list)} steps, language: {lang!r}")

    # Sample size estimate from 20 episodes
    print("\n=== Sample 20 episodes for step count estimate ===")
    sample_ds = tfds.load("taco_play", data_dir=str(DATA_ROOT), split="train", shuffle_files=False)
    sample_counts = []
    for ep in sample_ds.take(20):
        sample_counts.append(sum(1 for _ in ep["steps"]))
    mean_len = sum(sample_counts) / len(sample_counts)
    total_steps_est = int(mean_len * info.splits["train"].num_examples)
    print(f"  20-episode mean length: {mean_len:.1f} steps")
    print(f"  estimated total steps: {total_steps_est:,} "
          f"(= {mean_len:.1f} × {info.splits['train'].num_examples})")

    # Visualize first episode: scene + wrist frames
    print("\n=== Saving episode visualizations ===")
    for ep_idx, episode in enumerate(ds.take(3)):
        steps_list = list(episode["steps"].as_numpy_iterator())
        lang = ""
        for step in steps_list:
            for lang_key in ("language_instruction", "natural_language_instruction"):
                if lang_key in step and step[lang_key] is not None:
                    raw = step[lang_key]
                    if isinstance(raw, bytes):
                        lang = raw.decode("utf-8", errors="replace").strip()
                    else:
                        lang = str(raw).strip()
                    break
            if lang:
                break

        scene_frames = [s["observation"]["rgb_static"] for s in steps_list]
        wrist_frames = [s["observation"]["rgb_gripper"] for s in steps_list]
        print(f"  ep{ep_idx}: scene {scene_frames[0].shape}, wrist {wrist_frames[0].shape}, "
              f"{len(scene_frames)} frames")

        # Save frame 0 PNG
        Image.fromarray(scene_frames[0]).save(OUT_DIR / f"ep{ep_idx}_frame0_scene.png")
        Image.fromarray(wrist_frames[0]).save(OUT_DIR / f"ep{ep_idx}_frame0_wrist.png")

        # Side-by-side GIF (resize to common height 200)
        combined_frames = []
        target_h = 200
        for s, w in zip(scene_frames, wrist_frames):
            # Scene resize maintaining aspect
            s_img = Image.fromarray(s)
            s_w = int(s_img.width * target_h / s_img.height)
            s_img = s_img.resize((s_w, target_h), Image.LANCZOS)
            # Wrist resize (square, padded)
            w_img = Image.fromarray(w).resize((target_h, target_h), Image.LANCZOS)
            # Combine: scene | wrist
            canvas = Image.new("RGB", (s_w + 10 + target_h, target_h), (0, 0, 0))
            canvas.paste(s_img, (0, 0))
            canvas.paste(w_img, (s_w + 10, 0))
            combined_frames.append(canvas)

        # Save GIF (every 2nd frame to reduce size)
        gif_frames = combined_frames[::2]
        gif_path = OUT_DIR / f"ep{ep_idx}_scene_wrist.gif"
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=100,  # 10 fps
            loop=0,
            optimize=False,
        )
        print(f"    language: {lang!r}")
        print(f"    saved: ep{ep_idx}_frame0_{{scene,wrist}}.png, ep{ep_idx}_scene_wrist.gif "
              f"({len(gif_frames)} frames)")

    print(f"\nAll outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
