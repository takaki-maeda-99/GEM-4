"""
Phase 3a-2: per-dataset action / proprio statistics 計算 (Stage 1-2 dataset_statistics.json 互換 format).

対象 (Phase 3a-1 で download 済):
  - taco_play (Franka、3242 episodes、action.rel_actions_world 7-dim 直接取得)
  - fractal20220817_data (Google RT-1、87212 episodes、
      world_vector(3)+rotation_delta(3)+gripper_closedness_action(1) を concat して canonical 7-dim)

Canonical 7-dim action (全 dataset 共通、Stage 2 LIBERO と同構造):
  - dim 0-2: xyz delta (EEF linear、mask=True、q01/q99 で normalize)
  - dim 3-5: rotation delta (3-dim rot vec、mask=True、q01/q99 で normalize)
  - dim 6: gripper (mask=False、raw pass-through、post-process で binary 化)

Proprio は Stage 3 では不採用 (X-VLA 原実装に倣い pretrain では proprio 使わず、
fine-tune 時に再導入)、本 script では action stats のみ出力。

出力:
  data/stage3_openx/<name>/dataset_statistics.json
    {"action": {"q01":..., "q99":..., "mean":..., "std":..., "min":..., "max":..., "mask":...},
     "num_transitions": ..., "num_trajectories": ...}

Run (CPU only、~1-2h):
  .venv-gemma4/bin/python scripts/stage3/compute_dataset_statistics.py 2>&1 | tee /tmp/stage3_stats.log
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")   # R18

import tensorflow_datasets as tfds

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = REPO_ROOT / "data" / "stage3_openx"
LOG_PATH = REPO_ROOT / "runs" / "gemma4" / "stage3_data" / "stats.log"


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def extract_canonical_action_taco_play(step) -> np.ndarray:
    """Taco Play: action.rel_actions_world (7 dim) をそのまま canonical として採用.
    dim: [xyz_delta_x, xyz_delta_y, xyz_delta_z, rot_delta_x, rot_delta_y, rot_delta_z, gripper]
    """
    return step["action"]["rel_actions_world"].numpy().astype(np.float32)


def extract_canonical_action_fractal(step) -> np.ndarray:
    """Fractal: world_vector(3) + rotation_delta(3) + gripper_closedness_action(1) を concat.
    canonical 7-dim に整形、Taco Play と semantic 一致 (xyz delta + rot delta + gripper)。
    """
    wv = step["action"]["world_vector"].numpy().astype(np.float32)        # (3,)
    rd = step["action"]["rotation_delta"].numpy().astype(np.float32)       # (3,)
    gc = step["action"]["gripper_closedness_action"].numpy().astype(np.float32)  # (1,)
    return np.concatenate([wv, rd, gc], axis=0)   # (7,)


EXTRACTORS = {
    "taco_play": extract_canonical_action_taco_play,
    "fractal20220817_data": extract_canonical_action_fractal,
}


def collect_actions(dataset_name: str, max_episodes: int = -1) -> tuple:
    """全 (or 先頭 max_episodes) episode を iter して canonical 7-dim action を NumPy 累積."""
    log(f"=== Collecting actions for {dataset_name} ===")
    t0 = time.time()
    ds = tfds.load(dataset_name, data_dir=str(DATA_ROOT), split="train")
    extractor = EXTRACTORS[dataset_name]

    all_actions = []   # list[(7,) np.ndarray]
    num_episodes = 0
    num_transitions = 0
    last_progress = time.time()

    for ep_idx, ep in enumerate(ds):
        if max_episodes > 0 and ep_idx >= max_episodes:
            break
        for step in ep["steps"]:
            a = extractor(step)
            if a.shape != (7,):
                raise RuntimeError(f"unexpected action shape: {a.shape} at episode {ep_idx}")
            all_actions.append(a)
            num_transitions += 1
        num_episodes += 1

        if time.time() - last_progress > 30:
            elapsed = time.time() - t0
            rate = num_transitions / max(elapsed, 1)
            log(f"  progress ep {num_episodes} / {num_transitions} transitions / "
                f"{rate:.0f} trans/s / {elapsed:.0f}s elapsed")
            last_progress = time.time()

    actions_arr = np.stack(all_actions, axis=0)   # (N, 7)
    elapsed = time.time() - t0
    log(f"  done: {num_episodes} episodes, {num_transitions} transitions in {elapsed/60:.1f} min")
    log(f"  actions array shape: {actions_arr.shape}, memory: {actions_arr.nbytes/1e6:.1f} MB")
    return actions_arr, num_episodes, num_transitions


def compute_stats(actions: np.ndarray) -> dict:
    """Canonical 7-dim action の per-dim stats、Stage 2 LIBERO format 互換."""
    # mask: dim 0-5 は normalize、dim 6 (gripper) は raw pass-through
    mask = [True] * 6 + [False]
    return {
        "q01": np.percentile(actions, 1.0, axis=0).tolist(),
        "q99": np.percentile(actions, 99.0, axis=0).tolist(),
        "mean": actions.mean(axis=0).tolist(),
        "std": actions.std(axis=0).tolist(),
        "min": actions.min(axis=0).tolist(),
        "max": actions.max(axis=0).tolist(),
        "mask": mask,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default="",
                        help="Single dataset (taco_play or fractal20220817_data)")
    parser.add_argument("--max_episodes", type=int, default=-1,
                        help="Subset for debug. -1 = full")
    parser.add_argument("--outdir", type=str, default=str(DATA_ROOT))
    args = parser.parse_args()

    log("Stage 3 Phase 3a-2 start: per-dataset action statistics")
    log(f"  max_episodes: {args.max_episodes} (full=-1)")
    log(f"  outdir: {args.outdir}")

    datasets = ["taco_play", "fractal20220817_data"]
    if args.only:
        datasets = [args.only]

    all_stats = {}
    for ds_name in datasets:
        actions, num_ep, num_t = collect_actions(ds_name, max_episodes=args.max_episodes)
        stats = compute_stats(actions)
        payload = {
            ds_name: {
                "action": stats,
                "num_transitions": num_t,
                "num_trajectories": num_ep,
                "canonical_action_schema": "7-dim = xyz_delta(3) + rot_delta(3) + gripper(1)、mask=[T,T,T,T,T,T,F]",
                "source_action_fields": {
                    "taco_play": "action.rel_actions_world (7 dim 直接)",
                    "fractal20220817_data": "concat(world_vector(3), rotation_delta(3), gripper_closedness_action(1))",
                }.get(ds_name, "unknown"),
            }
        }
        out_path = Path(args.outdir) / ds_name / "dataset_statistics.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        log(f"  Wrote {out_path}")
        log(f"  Sample stats (dim 0-6):")
        for i in range(7):
            dim_name = ["Δx", "Δy", "Δz", "Δrx", "Δry", "Δrz", "gripper"][i]
            log(f"    {i} ({dim_name}): q01={stats['q01'][i]:+.4f}  q99={stats['q99'][i]:+.4f}  "
                f"mean={stats['mean'][i]:+.4f}  std={stats['std'][i]:.4f}  mask={stats['mask'][i]}")
        all_stats[ds_name] = payload[ds_name]

    # Also write combined stats for multi-dataset loader convenience
    combined = {"datasets": all_stats}
    combined_path = Path(args.outdir) / "combined_dataset_statistics.json"
    combined_path.write_text(json.dumps(combined, indent=2))
    log(f"\nCombined: {combined_path}")
    log("Stage 3 Phase 3a-2 done")


if __name__ == "__main__":
    main()
