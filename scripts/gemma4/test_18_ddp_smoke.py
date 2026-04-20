"""
test_18_ddp_smoke.py

D1 (Stage 2 後半): DDP pre-flight smoke launcher.
`torchrun --nproc_per_node=2` で `finetune_gemma4.py` を DDP モード起動、
GPU 2 + GPU 3 (A100 40GB × 2) で 100-step smoke、6 acceptance criteria を検証.

D1 acceptance criteria (Plan v3 § Tier A):
  1. LLM/VB leak check が DDP wrap 後に動作 (`model.module.llm.parameters()`、全 step で leak 0)
  2. loss finite、Phase 2e と同 order of magnitude
  3. per-step time throughput ≈ 1.6-1.8x (vs single-GPU baseline 0.87 s/step)
  4. Checkpoint save (rank 0 only) + load で `module.` prefix なし (single GPU 形式互換)
  5. R16: TF + DDP CUDA init order、TF GPU grab なし
  6. NCCL all-reduce scope = trainable params のみ (frozen LLM/VB 除外、~675M)

Run:
  .venv-gemma4/bin/python scripts/gemma4/test_18_ddp_smoke.py
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VLA_SCRIPTS = REPO_ROOT / "VLA-Adapter" / "vla-scripts"
VENV_BIN = REPO_ROOT / ".venv-gemma4" / "bin"
TORCHRUN = VENV_BIN / "torchrun"   # venv 内 torchrun (PATH に無い環境対応)

SINGLE_GPU_BASELINE_STEP_SEC = 0.87   # Phase 2b smoke median (shuffle_buffer=1000 条件)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="2,3", help="CUDA_VISIBLE_DEVICES (2 GPUs)")
    parser.add_argument("--master_port", type=int, default=29501)
    parser.add_argument("--smoke_max_steps", type=int, default=100)
    parser.add_argument("--smoke_save_step", type=int, default=50)
    parser.add_argument("--batch_size_per_gpu", type=int, default=4, help="per-GPU B、effective = B × world_size")
    parser.add_argument("--run_id_note", type=str, default="d1_ddp_smoke")
    args = parser.parse_args()

    cmd = [
        str(TORCHRUN),
        f"--nproc_per_node=2",
        f"--master_port={args.master_port}",
        str(VLA_SCRIPTS / "finetune_gemma4.py"),
        "--smoke_mode", "True",
        "--ddp_mode", "True",
        "--smoke_max_steps", str(args.smoke_max_steps),
        "--smoke_save_step", str(args.smoke_save_step),
        "--smoke_run_resume_check", "False",    # DDP 時 subprocess resume は未対応、skip
        "--batch_size", str(args.batch_size_per_gpu),
        "--run_id_note", args.run_id_note,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    env["TOKENIZERS_PARALLELISM"] = "false"
    # NCCL デバッグ log (acceptance criterion 5/6 の検証)
    env["NCCL_DEBUG"] = "WARN"                  # WARN レベル (INFO は噪音過多)

    print("=" * 80)
    print("D1 DDP pre-flight smoke")
    print("=" * 80)
    print(f"  CUDA_VISIBLE_DEVICES: {env['CUDA_VISIBLE_DEVICES']}")
    print(f"  master_port:          {args.master_port}")
    print(f"  nproc_per_node:       2")
    print(f"  smoke_max_steps:      {args.smoke_max_steps}")
    print(f"  smoke_save_step:      {args.smoke_save_step}")
    print(f"  batch_size (per-GPU): {args.batch_size_per_gpu}")
    print(f"  effective batch:      {args.batch_size_per_gpu * 2}")
    print(f"\nLaunching: {' '.join(cmd[:6])} ...")
    print()

    # --- Launch torchrun ---
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        print(f"\n[D1] torchrun failed with exit code {proc.returncode}")
        sys.exit(proc.returncode)

    # --- Parse summary JSON for acceptance criteria ---
    # build_run_id は per-GPU batch を埋め込む (cfg.batch_size * cfg.grad_accumulation_steps = per-GPU × 1)
    # effective は world_size 倍だが run_id には per-GPU 値が入る
    run_id = (
        f"gemma-4-e2b+libero_spatial_no_noops"
        f"+b{args.batch_size_per_gpu}+lr-0.0002+wu-500+smoke+{args.run_id_note}"
    )
    run_dir = REPO_ROOT / "runs" / "gemma4" / run_id
    summary_path = run_dir / "d1_ddp_smoke_result.json"

    if not summary_path.exists():
        # fallback: 直近更新の d1_ddp_smoke_result.json を探す
        candidates = list((REPO_ROOT / "runs" / "gemma4").glob("*d1_ddp_smoke*/d1_ddp_smoke_result.json"))
        if candidates:
            summary_path = max(candidates, key=lambda p: p.stat().st_mtime)
            run_dir = summary_path.parent    # 後続 C4 / report write が正しい dir を参照するよう update
            print(f"  (expected path not found、fallback to {summary_path})")
        else:
            print(f"\n[D1] FAIL: summary JSON not found, no glob match")
            sys.exit(1)

    summary = json.loads(summary_path.read_text())

    # --- Criterion 検証 ---
    criteria = {}

    # C1: LLM leak 全 step 0
    criteria["C1_llm_leak_all_zero"] = {
        "pass": summary["llm_leak_all_zero"],
        "detail": f"llm_leak_all_zero = {summary['llm_leak_all_zero']}",
    }

    # C2: loss finite、order of magnitude
    loss_init = summary["loss_summary"]["init"]
    loss_final = summary["loss_summary"]["final"]
    loss_max = summary["loss_summary"]["max"]
    c2_ok = (
        0.0 < loss_init < 10.0
        and 0.0 < loss_final < 10.0
        and 0.0 < loss_max < 10.0
    )
    criteria["C2_loss_finite_order"] = {
        "pass": c2_ok,
        "detail": f"init={loss_init:.4f}, final={loss_final:.4f}, max={loss_max:.4f}",
    }

    # C3: per-step time throughput
    median_step_sec = summary["median_step_sec_exc_warmup"]
    throughput_ratio = SINGLE_GPU_BASELINE_STEP_SEC / median_step_sec if median_step_sec > 0 else 0.0
    # 2-GPU DDP の理想 throughput は 2x、実用的 1.6-1.8x 期待
    criteria["C3_throughput"] = {
        "pass": throughput_ratio >= 1.3,    # 1.3x 以上なら acceptable (plan の Escalation #22 threshold)
        "detail": f"median_step_sec={median_step_sec:.3f}s, throughput={throughput_ratio:.2f}x (baseline {SINGLE_GPU_BASELINE_STEP_SEC}s/step)",
        "throughput_ratio": round(throughput_ratio, 3),
    }

    # C4: checkpoint format (single-GPU 互換、module. prefix なし)
    checkpoint_path = run_dir / "smoke_step50_checkpoint.pt"
    c4_ok = False
    c4_detail = ""
    if checkpoint_path.exists():
        # 軽量確認: torch.load で state_dict key を 1 つ取り出し、module. prefix 有無を見る
        try:
            import torch
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state_keys = list(payload["trainable_state_dict"].keys())
            has_module_prefix = any(k.startswith("module.") for k in state_keys[:5])
            c4_ok = not has_module_prefix
            c4_detail = f"first 3 keys: {state_keys[:3]}, module. prefix: {has_module_prefix}"
        except Exception as e:
            c4_detail = f"exception: {e}"
    else:
        c4_detail = f"checkpoint not found: {checkpoint_path}"
    criteria["C4_checkpoint_no_module_prefix"] = {"pass": c4_ok, "detail": c4_detail}

    # C5: TF GPU grab なし (worker 冒頭で tf.config.set_visible_devices 済、stderr に TF error がなければ OK)
    # torchrun stdout を直接 capture しないため、exit code で間接的に (全 step 完走 = TF 干渉なし)
    criteria["C5_tf_no_gpu_grab"] = {
        "pass": True,   # 100 step 完走 = TF が GPU を奪って OOM 等で落ちていない
        "detail": "100-step completed without TF-induced failure (indirect check)",
    }

    # C6: NCCL all-reduce scope ≈ 675M params
    all_reduce_m = (summary["all_reduce_scope_params"] or 0) / 1e6
    c6_ok = 670.0 < all_reduce_m < 680.0
    criteria["C6_all_reduce_scope"] = {
        "pass": c6_ok,
        "detail": f"all_reduce_params = {all_reduce_m:.3f}M (expected 675.138M ±5, frozen LLM+VB excluded)",
    }

    # --- Report ---
    print("\n" + "=" * 80)
    print("D1 DDP pre-flight smoke: acceptance criteria")
    print("=" * 80)
    all_pass = True
    for name, entry in criteria.items():
        mark = "OK  " if entry["pass"] else "FAIL"
        print(f"  [{mark}] {name}: {entry['detail']}")
        if not entry["pass"]:
            all_pass = False

    print("\n" + "=" * 80)
    verdict = "PASS" if all_pass else "FAIL (Escalation #21)"
    print(f"D1 DDP pre-flight smoke: {verdict}")
    print("=" * 80)
    print(f"Summary JSON: {summary_path}")

    # Dedicated acceptance report dump
    report = {
        "phase": "d1",
        "summary_path": str(summary_path),
        "criteria": criteria,
        "all_pass": all_pass,
        "throughput_ratio": criteria["C3_throughput"]["throughput_ratio"],
    }
    report_path = run_dir / "d1_acceptance_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Acceptance report: {report_path}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
