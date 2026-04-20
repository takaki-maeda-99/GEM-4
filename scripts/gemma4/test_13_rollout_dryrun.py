"""
test_13_rollout_dryrun.py

Phase 2c rollout dry-run: random-init `VLAAdapterGemma4` で LIBERO-Spatial 1 task × 1 episode 完走確認.

目的 (Plan v2 §5 Phase 2c、Check-in #12 提示物):
  - Stage 1 で未検証の全経路 (simulator init / action sampling / denormalize / env step) を **random-init model** で通過確認
  - 1 episode の完走 (success/fail 不問)
  - action 出力 shape `(NUM_ACTIONS_CHUNK, ACTION_DIM) = (8, 7)` の確認
  - denormalize 後の action range が LIBERO action space と整合 (position delta ≈ [-1, 1]、gripper [0, 1] → post-process {-1, +1})
  - observation 経路 (LIBERO 256x256 → resize 224 → image_transform → model) のエラーなし
  - model query 時間実測 (Phase 2e 本番 eval の per-step cost 推定用)

Exit criteria (Plan Escalation #3, #4 に該当しないこと):
  - simulator init 成功
  - checkpoint load 経路通過 (random-init でも load_model_state の empty path 分岐を検証)
  - action 常時 saturate (max/min 張り付き) **していない** (random-init model は無意味 action だが、output が単一値でない)
  - 1 episode が raise せず完走 (env.step が 220 step まで動く or task done で break)

Run: CUDA_VISIBLE_DEVICES=0 .venv-gemma4/bin/python scripts/gemma4/test_13_rollout_dryrun.py
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SCRIPTS_GEMMA4 = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_GEMMA4))

from eval_libero_gemma4 import EvalConfig, evaluate  # noqa: E402


def main():
    # --- Phase 2c dry-run config ---
    cfg = EvalConfig(
        checkpoint_path="",                    # random-init (Phase 2c 趣旨)
        task_suite_name="libero_spatial",
        task_id=0,                             # 1 task 固定
        num_tasks_limit=1,                     # 1 task のみ
        num_trials_per_task=1,                 # 1 episode のみ
        output_dir="runs/gemma4/eval",
        run_id_note="phase2c_dryrun",
        save_video=True,
        seed=7,
    )

    print("=" * 80)
    print("Phase 2c rollout dry-run (random-init VLAAdapterGemma4)")
    print("=" * 80)

    summary = evaluate(cfg)

    # ====================================================
    # Phase 2c Exit Criteria 判定
    # ====================================================
    eps = summary["episodes"]
    assert len(eps) == 1, f"expected 1 episode, got {len(eps)}"
    ep = eps[0]

    # Check 1: no exception in episode
    no_error = ep["error"] is None
    # Check 2: at least 1 model query (= action sampling 経路通過)
    queried = ep["num_model_queries"] > 0
    # Check 3: action shape returned correctly (derive from range_per_dim length)
    action_dim_ok = len(ep["action_denorm_range_per_dim"]) == 7
    # Check 4: action not constant (at least one dim has std > 0)
    non_saturate = any(s > 1e-6 for s in ep["action_denorm_std_per_dim"])
    # Check 5: proprio extracted per step
    proprio_ok = len(ep["proprio_range_per_dim"]) == 8
    # Check 6: replay frames captured
    replay_ok = ep["replay_n_frames"] > 0
    # Check 7: env step advanced (t > num_steps_wait)
    steps_advanced = ep["num_env_steps"] > 10   # cfg.num_steps_wait default

    passes = {
        "no_exception": no_error,
        "model_queried": queried,
        "action_dim_correct": action_dim_ok,
        "action_non_saturate": non_saturate,
        "proprio_extracted": proprio_ok,
        "replay_frames_captured": replay_ok,
        "env_step_advanced": steps_advanced,
    }

    print("\n" + "=" * 80)
    print("Phase 2c Exit Criteria")
    print("=" * 80)
    for k, v in passes.items():
        print(f"  {'OK ' if v else 'FAIL'}  {k}")
    all_pass = all(passes.values())

    print("\n" + "-" * 80)
    print("Episode diagnostic (Check-in #12 提示物)")
    print("-" * 80)
    print(f"  success:               {ep['success']}  (dry-run は不問、random-init で期待値 False)")
    print(f"  num_env_steps:         {ep['num_env_steps']}")
    print(f"  num_model_queries:     {ep['num_model_queries']}")
    print(f"  model_query_median_s:  {ep['model_query_median_s']:.3f}")
    print(f"  model_query_total_s:   {ep['model_query_total_s']:.2f}")
    print(f"  episode_wall_sec:      {ep['episode_wall_sec']}")
    print(f"  replay_n_frames:       {ep['replay_n_frames']}")
    print(f"  task_description:      {ep['task_description']!r}")

    print("\n  action_denorm_range_per_dim (7 dim):")
    for d, (lo, hi) in enumerate(ep["action_denorm_range_per_dim"]):
        mean = ep["action_denorm_mean_per_dim"][d]
        std = ep["action_denorm_std_per_dim"][d]
        dim_name = ["Δx", "Δy", "Δz", "Δrx", "Δry", "Δrz", "gripper"][d]
        print(f"    dim {d} ({dim_name:<8s}): [{lo:+.3f}, {hi:+.3f}]  mean={mean:+.3f}  std={std:.3f}")

    print("\n  proprio_range_per_dim (8 dim、raw/unnormalized):")
    prop_names = ["eef_x", "eef_y", "eef_z", "rot1", "rot2", "rot3", "grip1", "grip2"]
    for d, (lo, hi) in enumerate(ep["proprio_range_per_dim"]):
        print(f"    dim {d} ({prop_names[d]:<8s}): [{lo:+.3f}, {hi:+.3f}]")

    print("\n  action_stats (q01/q99/mask from dataset_statistics.json):")
    a_stats = summary["action_stats"]
    for d in range(7):
        dim_name = ["Δx", "Δy", "Δz", "Δrx", "Δry", "Δrz", "gripper"][d]
        print(f"    dim {d} ({dim_name:<8s}): q01={a_stats['q01'][d]:+.3f}  q99={a_stats['q99'][d]:+.3f}  mask={a_stats['mask'][d]}")

    print("\n" + "=" * 80)
    print(f"Phase 2c dry-run: {'PASS' if all_pass else 'FAIL (要 User 判断)'}")
    print("=" * 80)

    # Dedicated dry-run summary JSON
    dryrun_summary = {
        "phase": "2c.dryrun",
        "exit_criteria": passes,
        "all_pass": all_pass,
        "episode": ep,
        "action_stats_source": summary["action_stats"],
        "checkpoint_load_info": summary["checkpoint_load_info"],
    }
    out_path = Path("runs/gemma4/eval/libero_spatial--phase2c_dryrun") / "test_13_dryrun_result.json"
    if out_path.parent.exists():
        out_path.write_text(json.dumps(dryrun_summary, indent=2, default=str))
        print(f"\nDry-run summary JSON: {out_path}")


if __name__ == "__main__":
    main()
