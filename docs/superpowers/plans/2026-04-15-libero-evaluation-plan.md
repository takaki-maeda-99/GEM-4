# LIBERO Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate VLA-Gemma4 on the LIBERO libero_spatial benchmark, measuring task success rate over 100 episodes (10 tasks × 10 episodes).

**Architecture:** Self-contained evaluation script that loads a trained checkpoint, creates LIBERO simulation environments, and runs a predict→step loop to measure success rate. Uses existing VLAPolicy.predict() with no changes to core model code.

**Tech Stack:** LIBERO (lerobot-libero fork), robosuite, MuJoCo, existing VLA-Gemma4 stack

**Spec:** `docs/superpowers/specs/2026-04-15-libero-evaluation-design.md`

---

## File Structure

```
vla_gemma4/
└── configs/
    └── libero_spatial.yaml     # LIBERO-specific config (7DOF, 9D proprio, 2 cameras)

scripts/
└── eval_libero.py              # LIBERO sim evaluation script
```

No existing files are modified.

---

## Task 1: LIBERO Config File

**Files:**
- Create: `vla_gemma4/configs/libero_spatial.yaml`

- [ ] **Step 1: Create the config file**

`vla_gemma4/configs/libero_spatial.yaml`:
```yaml
# LIBERO libero_spatial benchmark config
# 7DOF EEF delta actions, 9D proprio, 2 cameras

model_name: "google/gemma-4-E2B-it"
num_action_tokens: 1
visual_token_budget: 140

action_dim: 7
chunk_size: 1
proprio_dim: 9
proprio_key: "observation.state"

# Evaluation: LIBERO observation key names
cameras:
  - "agentview_image"
  - "robot0_eye_in_hand_image"

# Training: LeRobot dataset camera key names
lerobot_cameras:
  - "observation.images.agentview"
  - "observation.images.eye_in_hand"

action_head:
  type: "mlp"
  hidden_dims: [512, 256]
  gripper_as_binary: true

training:
  strategy: "lora"
  lora:
    r: 8
    alpha: 16
    target_modules: ["q_proj", "v_proj"]
  batch_size: 16
  lr: 1.0e-4
  weight_decay: 0.01
  num_epochs: 5
  warmup_steps: 100
  max_grad_norm: 1.0
  mixed_precision: "bf16"
  save_every_n_steps: 500
  eval_every_n_steps: 250

data:
  dataset_name: "HuggingFaceVLA/libero_spatial"
  language_instruction_key: "language_instruction"
  default_instruction: "manipulation task"
  normalizer_path: null

inference:
  quantization: "4bit"
```

- [ ] **Step 2: Commit**

```bash
git add vla_gemma4/configs/libero_spatial.yaml
git commit -m "feat: add LIBERO libero_spatial config"
```

---

## Task 2: LIBERO Evaluation Script

**Files:**
- Create: `scripts/eval_libero.py`

- [ ] **Step 1: Verify LIBERO imports work**

Run:
```bash
MUJOCO_GL=egl uv run python3 -c "
from libero.libero.benchmark import get_benchmark
b = get_benchmark('libero_spatial')()
print(f'Tasks: {b.get_num_tasks()}')
print(f'Task 0: {b.get_task(0).name}')
"
```
Expected: `Tasks: 10` and a task name printed.

- [ ] **Step 2: Verify LIBERO env can be created**

Run:
```bash
MUJOCO_GL=egl uv run python3 -c "
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv
import numpy as np

b = get_benchmark('libero_spatial')()
bddl = b.get_task_bddl_file_path(0)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
obs = env.reset()
print('Obs keys:', sorted(obs.keys()))
for k, v in obs.items():
    if hasattr(v, 'shape'): print(f'  {k}: {v.shape} {v.dtype}')
obs, r, d, info = env.step(np.zeros(7))
print(f'Step OK, reward={r}, done={d}, success={info.get(\"success\", False)}')
env.close()
print('Done')
"
```
Expected: obs keys printed, step runs without error.

- [ ] **Step 3: Write the evaluation script**

`scripts/eval_libero.py`:
```python
"""LIBERO benchmark evaluation for VLA-Gemma4."""

import argparse
import json
import logging
import os

import numpy as np
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def convert_obs_to_batch(obs: dict, instruction: str, camera_keys: list[str]) -> dict:
    """Convert LIBERO observation dict to VLAPolicy batch format."""
    images = []
    for cam_key in camera_keys:
        img = torch.from_numpy(obs[cam_key]).permute(2, 0, 1).float() / 255.0
        images.append(img.unsqueeze(0))  # [1, C, H, W]

    proprio = np.concatenate([
        obs["robot0_eef_pos"],       # (3,)
        obs["robot0_eef_quat"],      # (4,)
        obs["robot0_gripper_qpos"],  # (2,)
    ])

    return {
        "images": images,
        "instruction": [instruction],
        "proprio": torch.from_numpy(proprio).float().unsqueeze(0),  # [1, 9]
    }


def evaluate_libero(
    policy,
    suite_name: str,
    camera_keys: list[str],
    n_episodes: int = 10,
    max_steps: int = 300,
    seed: int = 42,
) -> dict:
    """Run LIBERO evaluation and return results."""
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    benchmark = get_benchmark(suite_name)()
    num_tasks = benchmark.get_num_tasks()
    logger.info(f"Suite: {suite_name}, {num_tasks} tasks, {n_episodes} episodes each")

    np.random.seed(seed)
    results = {}

    for task_idx in range(num_tasks):
        task = benchmark.get_task(task_idx)
        bddl_path = benchmark.get_task_bddl_file_path(task_idx)
        init_states = benchmark.get_task_init_states(task_idx)

        logger.info(f"Task {task_idx}/{num_tasks}: {task.name}")

        env = OffScreenRenderEnv(
            bddl_file_name=bddl_path,
            camera_heights=128,
            camera_widths=128,
        )
        successes = 0

        for episode in range(n_episodes):
            env.reset()
            obs = env.set_init_state(init_states[episode])

            # Settle physics with zero actions
            for _ in range(5):
                obs, _, _, _ = env.step(np.zeros(7))

            for step in range(max_steps):
                batch = convert_obs_to_batch(obs, task.language, camera_keys)
                with torch.no_grad():
                    action = policy.predict(batch)  # [1, 1, 7]
                action_np = action[0, 0].cpu().numpy()
                obs, reward, done, info = env.step(action_np)

                if info.get("success", False):
                    break

            success = int(info.get("success", False))
            successes += success
            logger.info(f"  Episode {episode}: {'SUCCESS' if success else 'FAIL'} (step {step})")

        env.close()
        task_sr = successes / n_episodes
        results[task.name] = task_sr
        logger.info(f"  Task success rate: {task_sr:.1%}")

    overall_sr = sum(results.values()) / len(results)
    logger.info(f"\nOverall success rate: {overall_sr:.1%}")

    return {
        "suite": suite_name,
        "n_episodes_per_task": n_episodes,
        "max_steps": max_steps,
        "seed": seed,
        "tasks": results,
        "overall_success_rate": overall_sr,
    }


def main():
    parser = argparse.ArgumentParser(description="LIBERO benchmark evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--suite", type=str, default="libero_spatial")
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="libero_results.json")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")

    config = load_config(args.config)
    camera_keys = config["cameras"]

    # Build policy and load checkpoint
    from vla_gemma4.scripts.train import build_policy, apply_lora

    logger.info("Loading model...")
    policy = build_policy(config)

    if config["training"]["strategy"] == "lora":
        policy = apply_lora(policy, config)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_state = checkpoint["model_state_dict"]
    current_state = policy.state_dict()
    filtered_state = {k: v for k, v in saved_state.items() if k in current_state}
    policy.load_state_dict(filtered_state, strict=False)
    logger.info(f"Loaded {len(filtered_state)} / {len(saved_state)} keys from checkpoint")

    del checkpoint, saved_state
    torch.cuda.empty_cache()

    # Run evaluation
    results = evaluate_libero(
        policy=policy,
        suite_name=args.suite,
        camera_keys=camera_keys,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Smoke test the script parses**

Run:
```bash
uv run python3 -c "import scripts.eval_libero; print('import OK')"
```
Expected: No import errors.

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_libero.py
git commit -m "feat: add LIBERO benchmark evaluation script"
```

---

## Task 3: End-to-End Verification

This task verifies the full pipeline works: env creation → obs conversion → policy predict → env step.

- [ ] **Step 1: Run eval_libero on 1 task, 1 episode as smoke test**

Run:
```bash
MUJOCO_GL=egl uv run python3 scripts/eval_libero.py \
  --config vla_gemma4/configs/libero_spatial.yaml \
  --checkpoint outputs/checkpoint_step_500.pt \
  --suite libero_spatial \
  --n_episodes 1 \
  --max_steps 10 \
  --output outputs/libero_smoke_test.json
```

Expected: Script runs to completion, outputs JSON with 10 task results (most will be 0.0 success rate since only 1 episode with 10 steps).

- [ ] **Step 2: Verify output JSON format**

Run:
```bash
cat outputs/libero_smoke_test.json | python3 -m json.tool
```

Expected: Valid JSON with `suite`, `tasks` dict, and `overall_success_rate`.

- [ ] **Step 3: Commit any fixes needed**

```bash
git add -A
git commit -m "fix: eval_libero smoke test fixes"
```
