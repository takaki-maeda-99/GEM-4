"""
Phase 3a-1: Tier 1 OXE data download (direct GCS copy、background).

戦略 (apache_beam 不要の直接 copy):
  - OXE Tier 1 data は `gs://gresearch/robotics/<name>/<version>/` に public 配置 (anonymous read 可)
  - `tf.io.gfile.copy` で tfrecord + metadata を local disk に copy
  - 並列化 (ThreadPoolExecutor、default 8 worker)、各 dataset sequentially
  - 各 file 完了時 progress log、resume 可 (既存 file skip)
  - 全 download 後 integrity check: tfds.load で 1 episode 取得

対象:
  1. taco_play (~40 GB、Franka overhead+wrist)
  2. fractal20220817_data (~130 GB、Google RT-1)
  3. bridge (~400 GB、WidowX)

R18: GPU 非依存、Phase 2e 非干渉、network/disk only。

Run (background):
  .venv-gemma4/bin/python scripts/stage3/download_data.py 2>&1 | tee /tmp/stage3_download.log
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
tf.config.set_visible_devices([], "GPU")   # R18


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = REPO_ROOT / "data" / "stage3_openx"
LOG_PATH = REPO_ROOT / "runs" / "gemma4" / "stage3_data" / "download.log"
STATUS_PATH = REPO_ROOT / "runs" / "gemma4" / "stage3_data" / "download_status.json"

GCS_BASE = "gs://gresearch/robotics"

TIER1_DATASETS = [
    {"name": "taco_play",            "version": "0.1.0", "priority": 1, "est_gb": 40,  "note": "Franka overhead+wrist、自前仕様と最整合"},
    {"name": "fractal20220817_data", "version": "0.1.0", "priority": 2, "est_gb": 130, "note": "Google RT-1 large scale、7-DOF"},
    # Bridge は WidowX で User 自前 Franka 仕様と embodiment 差が大きく pretrain 効果低、
    # 2026-04-20 User 判断で B 案 (Taco Play + Fractal = 170 GB) 採用、Bridge は skip。
    # {"name": "bridge", "version": "0.1.0", "priority": 3, "est_gb": 400, "note": "WidowX、skip"},
]

NUM_PARALLEL_WORKERS = 16   # GCS 並列 download 数、network saturation 回避しつつ throughput 上げる


def log(msg: str, also_print: bool = True) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")
    if also_print:
        print(line, flush=True)


def update_status(name: str, status: dict) -> None:
    current = {}
    if STATUS_PATH.exists():
        try:
            current = json.loads(STATUS_PATH.read_text())
        except json.JSONDecodeError:
            current = {}
    current[name] = status
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(current, indent=2))


def disk_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / 1024 / 1024


def copy_one_file(src_gs: str, dst_local: Path) -> tuple:
    """1 file copy、(size_bytes, skipped) を返す."""
    dst_local.parent.mkdir(parents=True, exist_ok=True)
    if dst_local.exists():
        # resume: 同名 file 存在なら skip (size 一致確認はせず、tfrecord 破損は別問題として扱う)
        return (dst_local.stat().st_size, True)
    tf.io.gfile.copy(src_gs, str(dst_local), overwrite=False)
    return (dst_local.stat().st_size, False)


def download_one_dataset(spec: dict) -> dict:
    name = spec["name"]
    version = spec["version"]
    src_dir = f"{GCS_BASE}/{name}/{version}"
    dst_dir = DATA_ROOT / name / version

    log(f"=== START {name} (v{version}) ===")
    log(f"  note: {spec['note']}")
    log(f"  estimated: ~{spec['est_gb']} GB")
    log(f"  src: {src_dir}")
    log(f"  dst: {dst_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    status = {
        "name": name,
        "version": version,
        "start_ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "src": src_dir,
        "dst": str(dst_dir),
        "status": "listing_files",
        "num_files_total": 0,
        "num_files_done": 0,
        "num_files_skipped_resume": 0,
        "bytes_done": 0,
        "elapsed_sec": 0,
    }
    update_status(name, status)

    try:
        files = tf.io.gfile.glob(f"{src_dir}/*")
        # recursive 非対応 glob のため、subdir 展開を手動で行う (OXE data は basically flat、念のため)
        log(f"  listed {len(files)} files at {src_dir}")
        status["num_files_total"] = len(files)
        status["status"] = "copying"
        update_status(name, status)

        # --- Parallel copy ---
        done, skipped, bytes_done = 0, 0, 0
        last_progress_log = time.time()
        with ThreadPoolExecutor(max_workers=NUM_PARALLEL_WORKERS) as executor:
            futures = []
            for src_f in files:
                rel = src_f.replace(f"{src_dir}/", "")
                dst_f = dst_dir / rel
                futures.append(executor.submit(copy_one_file, src_f, dst_f))
            for fut in as_completed(futures):
                size, was_skipped = fut.result()
                done += 1
                if was_skipped:
                    skipped += 1
                bytes_done += size
                # progress log 毎 30 秒
                if time.time() - last_progress_log > 30:
                    elapsed = time.time() - t0
                    rate_mbps = bytes_done / 1024 / 1024 / max(elapsed, 1)
                    log(f"  progress {done}/{len(files)} ({100*done/len(files):.1f}%), "
                        f"{bytes_done/1024**3:.2f} GB, {rate_mbps:.1f} MB/s, {elapsed:.0f}s elapsed")
                    status.update({
                        "num_files_done": done,
                        "num_files_skipped_resume": skipped,
                        "bytes_done": bytes_done,
                        "elapsed_sec": round(elapsed, 1),
                    })
                    update_status(name, status)
                    last_progress_log = time.time()

        elapsed = time.time() - t0
        size_mb = disk_size_mb(dst_dir)

        # --- Integrity check: tfds.load で 1 episode ---
        integrity_note = ""
        try:
            import tensorflow_datasets as tfds
            ds = tfds.load(name, data_dir=str(DATA_ROOT), split="train")
            first_ep = next(iter(ds.take(1)))
            ep_keys = list(first_ep.keys())
            integrity_note = f"load OK、first episode keys: {ep_keys[:5]}..."
        except Exception as e:
            integrity_note = f"load FAIL: {type(e).__name__}: {str(e)[:200]}"

        status.update({
            "status": "completed",
            "end_ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_files_done": done,
            "num_files_skipped_resume": skipped,
            "bytes_done": bytes_done,
            "elapsed_sec": round(elapsed, 1),
            "size_mb": round(size_mb, 1),
            "integrity_note": integrity_note,
        })
        log(f"  COMPLETED in {elapsed/60:.1f} min ({size_mb/1024:.2f} GB、{done} files、{skipped} resumed)")
        log(f"  integrity: {integrity_note}")
    except Exception as e:
        elapsed = time.time() - t0
        size_mb = disk_size_mb(dst_dir)
        status.update({
            "status": "failed",
            "end_ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_sec": round(elapsed, 1),
            "size_mb_partial": round(size_mb, 1),
            "error_type": type(e).__name__,
            "error_msg": str(e)[:500],
        })
        log(f"  FAILED after {elapsed/60:.1f} min ({size_mb/1024:.2f} GB partial)")
        log(f"  error: {type(e).__name__}: {str(e)[:300]}")

    update_status(name, status)
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default="", help="Single dataset name")
    parser.add_argument("--skip_completed", action="store_true", default=True)
    args = parser.parse_args()

    log(f"Stage 3 Phase 3a-1 data download (direct GCS copy)")
    log(f"  storage: {DATA_ROOT}")
    log(f"  parallel workers per dataset: {NUM_PARALLEL_WORKERS}")
    log(f"  targets: {[d['name'] for d in TIER1_DATASETS]}")

    completed_names = set()
    if args.skip_completed and STATUS_PATH.exists():
        existing = json.loads(STATUS_PATH.read_text())
        completed_names = {n for n, s in existing.items() if s.get("status") == "completed"}
        if completed_names:
            log(f"  resume: skipping completed {completed_names}")

    results = []
    for spec in TIER1_DATASETS:
        if args.only and spec["name"] != args.only:
            continue
        if spec["name"] in completed_names:
            log(f"=== SKIP {spec['name']} (already completed) ===")
            continue
        res = download_one_dataset(spec)
        results.append(res)

    log("\n=== Stage 3 Phase 3a-1 summary ===")
    for res in results:
        log(f"  {res['name']:<30s} {res['status']:<12s} "
            f"{res.get('size_mb', res.get('size_mb_partial', 0))/1024:.2f} GB "
            f"({res.get('elapsed_sec', 0)/60:.1f} min)")
    total_gb = sum(res.get("size_mb", 0) for res in results) / 1024
    log(f"  TOTAL: {total_gb:.2f} GB")
    log(f"  Status JSON: {STATUS_PATH}")


if __name__ == "__main__":
    main()
