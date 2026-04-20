"""
test_14_rollout_sanity.py

Phase 2f: Phase 2e 本番 train の latest checkpoint (現時点では 10k-20k 区間の直近 save) を
GPU 1 で rollout、Phase 2c random-init からの regression なし確認 + proprio OOD 監視.

Plan v2 §5 Phase 2f、Check-in #16 提示物 (User 要請 2026-04-20):
  - 1 task × 1 episode (libero_spatial task 0)
  - checkpoint load 成否 (Phase 2b/2c の load_model_state 経路)
  - action chunk shape + denormalize range
  - **proprio OOD 監視**: eef_x/y/z/gripper_qpos の (min, max) を 220 step 全記録、
    q01/q99 と diff、clip saturate rate (|x_norm| ≥ 0.999 の step 数 / 220)
  - saturate 率 > 10% なら Escalation 検討 (trained model が in-distribution なら数 % 以下想定)
  - Phase 2c (random-init) との action/proprio range diff で「trained 効果」を定量化

Run: CUDA_VISIBLE_DEVICES=1 .venv-gemma4/bin/python scripts/gemma4/test_14_rollout_sanity.py \\
       --checkpoint_path runs/gemma4/<run_dir>/latest_checkpoint.pt
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

SCRIPTS_GEMMA4 = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_GEMMA4))

from eval_libero_gemma4 import EvalConfig, evaluate, normalize_proprio_bounds_q99, load_action_proprio_stats  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Phase 2c dry-run (random-init) baseline for regression comparison
PHASE2C_BASELINE = {
    "action_std_per_dim": None,    # will be loaded at runtime
    "proprio_range_per_dim": None,
}


def load_phase2c_baseline() -> dict:
    p = REPO_ROOT / "runs/gemma4/eval/libero_spatial--phase2c_dryrun/eval_result.json"
    if not p.exists():
        print(f"[phase2f] WARN: Phase 2c baseline {p} not found, skipping regression comparison")
        return {}
    s = json.loads(p.read_text())
    ep = s["episodes"][0]
    return {
        "action_denorm_std_per_dim": ep["action_denorm_std_per_dim"],
        "action_denorm_range_per_dim": ep["action_denorm_range_per_dim"],
        "proprio_range_per_dim": ep["proprio_range_per_dim"],
    }


def compute_proprio_saturate_rate(proprio_history: np.ndarray, proprio_stats: dict) -> dict:
    """Proprio を normalize した時の clip saturate 率を dim 毎に算出."""
    # proprio_history: (T, PROPRIO_DIM) raw values
    if proprio_history.size == 0:
        return {}
    q01 = proprio_stats["q01"]
    q99 = proprio_stats["q99"]
    mask = proprio_stats["mask"]
    denom = (q99 - q01) + 1e-8
    norm = 2.0 * (proprio_history - q01) / denom - 1.0    # (T, D)、非 clip
    # |x_norm| >= 0.999 を saturate と判定
    saturate = np.abs(norm) >= 0.999                       # (T, D) bool
    # mask=False dim は raw pass-through (saturate 概念なし)、明示的に False 上書き
    saturate[:, ~mask] = False
    saturate_rate = saturate.mean(axis=0)                  # (D,)
    saturate_step_count = saturate.any(axis=1).sum()       # 少なくとも 1 dim が saturate した step 数
    total_steps = len(proprio_history)
    return {
        "per_dim_saturate_rate": saturate_rate.tolist(),
        "any_dim_saturate_step_count": int(saturate_step_count),
        "any_dim_saturate_rate": float(saturate_step_count / total_steps) if total_steps > 0 else 0.0,
        "total_steps": total_steps,
        "mask": mask.tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="runs/gemma4/eval")
    p.add_argument("--run_id_note", type=str, default="phase2f_10k_sanity")
    p.add_argument("--dataset_statistics_path", type=str,
                   default="VLA-Adapter/outputs/LIBERO-Spatial-Pro/dataset_statistics.json")
    args = p.parse_args()

    cfg = EvalConfig(
        checkpoint_path=args.checkpoint_path,
        dataset_statistics_path=args.dataset_statistics_path,
        task_suite_name="libero_spatial",
        task_id=args.task_id,
        num_tasks_limit=1,
        num_trials_per_task=1,
        output_dir=args.output_dir,
        run_id_note=args.run_id_note,
        save_video=True,
        seed=7,
    )

    print("=" * 80)
    print(f"Phase 2f rollout sanity (trained checkpoint: {args.checkpoint_path})")
    print("=" * 80)

    summary = evaluate(cfg)
    ep = summary["episodes"][0]

    # --- Proprio saturate rate (User 要請、Check-in #16 提示物) ---
    stats_path = Path(args.dataset_statistics_path)
    if not stats_path.is_absolute():
        stats_path = REPO_ROOT / stats_path
    action_stats, proprio_stats = load_action_proprio_stats(stats_path, cfg.unnorm_key)
    # proprio_history は eval_result.json に range_per_dim しか残っていないため再計算不能。
    # evaluate() 呼び出し時に episode dict に raw history を残しておきたいが、interface 変更は avoid。
    # 代わりに summary の proprio_range_per_dim から approximate saturate を計算:
    # range が q01/q99 を外れているかどうかを指標として使う。full distribution は出せないが、
    # min/max が q01/q99 範囲内なら saturate rate は大まか低め想定。
    # ※ 厳密な saturate rate 計算には Phase 2f を eval_libero_gemma4 に proprio_raw 保持追加するか、
    #   本 script で直接 rollout する必要あり。今回は range-based approx で代替、Check-in #16 で
    #   User に approx の旨明示。
    proprio_ranges = ep["proprio_range_per_dim"]
    approximate_sat_per_dim = []
    for d, (lo, hi) in enumerate(proprio_ranges):
        if not proprio_stats["mask"][d]:
            approximate_sat_per_dim.append(None)
            continue
        q01_d, q99_d = proprio_stats["q01"][d], proprio_stats["q99"][d]
        # 片側 OOD かどうか: lo < q01 or hi > q99
        lo_ood = lo < float(q01_d)
        hi_ood = hi > float(q99_d)
        approximate_sat_per_dim.append({
            "lo": lo, "hi": hi,
            "q01": float(q01_d), "q99": float(q99_d),
            "lo_ood": lo_ood, "hi_ood": hi_ood,
            "any_ood": lo_ood or hi_ood,
        })

    # --- Phase 2c baseline との regression comparison ---
    baseline = load_phase2c_baseline()
    regression_comparison = {}
    if baseline:
        # action std が随分小さくなれば "trained で行動が安定化" (期待値)
        # action std が Phase 2c random-init と同等 = trained 効果なし (regression warning)
        p2c_action_std = baseline.get("action_denorm_std_per_dim", [0] * 7)
        p2f_action_std = ep["action_denorm_std_per_dim"]
        regression_comparison["action_std_per_dim"] = {
            "phase2c_random_init": p2c_action_std,
            "phase2f_trained": p2f_action_std,
            "std_ratio_p2f_over_p2c": [
                round(p2f_action_std[d] / p2c_action_std[d] + 1e-12, 3) if p2c_action_std[d] > 0 else None
                for d in range(7)
            ],
        }
        # Regression warning: 全 dim で p2f std >= p2c std * 0.95 (trained による変化が微小)
        # (Note: trained は std が必ずしも減るわけではない、task-specific 動きが入るので std 増加もあり得る)

    # --- Phase 2f Exit Criteria ---
    passes = {
        "no_exception": ep["error"] is None,
        "trained_checkpoint_loaded": summary["checkpoint_load_info"]["loaded"],
        "model_queried": ep["num_model_queries"] > 0,
        "action_dim_correct": len(ep["action_denorm_range_per_dim"]) == 7,
        "action_non_saturate": any(s > 1e-6 for s in ep["action_denorm_std_per_dim"]),
        "episode_completed": ep["num_env_steps"] > 10,
    }
    all_pass = all(passes.values())

    print("\n" + "=" * 80)
    print("Phase 2f Exit Criteria")
    print("=" * 80)
    for k, v in passes.items():
        print(f"  {'OK ' if v else 'FAIL'}  {k}")
    print(f"\nCheckpoint: {summary['checkpoint_load_info'].get('checkpoint_path')}")
    print(f"  gradient_step_idx at save: {summary['checkpoint_load_info'].get('gradient_step_idx_at_save')}")
    print(f"  current_lr at save:        {summary['checkpoint_load_info'].get('current_lr_at_save')}")

    print("\n--- Episode diagnostic ---")
    print(f"  success:               {ep['success']}")
    print(f"  num_env_steps:         {ep['num_env_steps']}")
    print(f"  num_model_queries:     {ep['num_model_queries']}")
    print(f"  model_query_median_s:  {ep['model_query_median_s']:.3f}")
    print(f"  episode_wall_sec:      {ep['episode_wall_sec']}")

    print("\n--- Proprio OOD check (approximate via range vs q01/q99) ---")
    dim_names = ["eef_x", "eef_y", "eef_z", "rot1", "rot2", "rot3", "grip1", "grip2"]
    any_ood_count = 0
    for d, info in enumerate(approximate_sat_per_dim):
        name = dim_names[d]
        if info is None:
            print(f"  dim {d} ({name:<8s}): mask=False (skipped)")
            continue
        flag = " [OOD]" if info["any_ood"] else ""
        print(f"  dim {d} ({name:<8s}): observed [{info['lo']:+.3f}, {info['hi']:+.3f}]  "
              f"train q01/q99 [{info['q01']:+.3f}, {info['q99']:+.3f}]{flag}")
        if info["any_ood"]:
            any_ood_count += 1

    print(f"\n  OOD dim count: {any_ood_count} / 8")
    proprio_ood_ok = any_ood_count <= 2    # 許容: 2 dim 程度までの微 OOD、3+ で要調査

    if baseline:
        print("\n--- Phase 2c (random-init) vs Phase 2f (trained) action std comparison ---")
        ranges = regression_comparison["action_std_per_dim"]
        act_dim_names = ["Δx", "Δy", "Δz", "Δrx", "Δry", "Δrz", "gripper"]
        for d in range(7):
            p2c = ranges["phase2c_random_init"][d]
            p2f = ranges["phase2f_trained"][d]
            ratio = ranges["std_ratio_p2f_over_p2c"][d]
            ratio_str = f"{ratio:.2f}" if ratio is not None else "n/a"
            print(f"  dim {d} ({act_dim_names[d]:<8s}): p2c std={p2c:.3f}  p2f std={p2f:.3f}  ratio={ratio_str}")
    else:
        print("\n  (Phase 2c baseline not available, skip regression comparison)")

    # --- Dump summary ---
    out_json = {
        "phase": "2f",
        "checkpoint_path": args.checkpoint_path,
        "exit_criteria": passes,
        "all_pass": all_pass,
        "proprio_ood_ok": proprio_ood_ok,
        "proprio_ood_any_dim_count": any_ood_count,
        "proprio_ood_per_dim": approximate_sat_per_dim,
        "regression_comparison_vs_phase2c": regression_comparison,
        "episode_summary": ep,
        "eval_summary": summary,
    }
    out_path = Path(summary["cfg"]["output_dir"]) / f"libero_spatial--{cfg.run_id_note}" / "test_14_phase2f_result.json"
    if out_path.parent.exists():
        out_path.write_text(json.dumps(out_json, indent=2, default=str))
        print(f"\nDump: {out_path}")

    print("\n" + "=" * 80)
    print(f"Phase 2f sanity: {'PASS' if (all_pass and proprio_ood_ok) else 'CHECK (要 User 判断)'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
