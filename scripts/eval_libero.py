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
        images.append(img.unsqueeze(0))

    proprio = np.concatenate([
        obs["robot0_eef_pos"],
        obs["robot0_eef_quat"],
        obs["robot0_gripper_qpos"],
    ])

    return {
        "images": images,
        "instruction": [instruction],
        "proprio": torch.from_numpy(proprio).float().unsqueeze(0),
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
        # Load init states manually (torch.load needs weights_only=False for pickle data)
        init_states_path = os.path.join(
            bddl_path.replace("bddl_files", "init_states").rsplit("/", 1)[0],
            task.init_states_file,
        )
        init_states = torch.load(init_states_path, map_location="cpu", weights_only=False)

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

            for _ in range(5):
                obs, _, _, _ = env.step(np.zeros(7))

            for step in range(max_steps):
                batch = convert_obs_to_batch(obs, task.language, camera_keys)
                with torch.no_grad():
                    action = policy.predict(batch)
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
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path (omit for random weights)")
    parser.add_argument("--suite", type=str, default="libero_spatial")
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="libero_results.json")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")

    config = load_config(args.config)
    camera_keys = config["cameras"]

    from vla_gemma4.scripts.train import build_policy, apply_lora

    logger.info("Loading model...")
    policy = build_policy(config)

    if config["training"]["strategy"] == "lora":
        policy = apply_lora(policy, config)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        saved_state = checkpoint["model_state_dict"]
        current_state = policy.state_dict()
        filtered_state = {k: v for k, v in saved_state.items()
                          if k in current_state and saved_state[k].shape == current_state[k].shape}
        policy.load_state_dict(filtered_state, strict=False)
        logger.info(f"Loaded {len(filtered_state)} / {len(saved_state)} keys from checkpoint")
        del checkpoint, saved_state
        torch.cuda.empty_cache()
    else:
        logger.info("No checkpoint specified, using random weights")

    results = evaluate_libero(
        policy=policy,
        suite_name=args.suite,
        camera_keys=camera_keys,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
