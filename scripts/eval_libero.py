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

    # LeRobot dataset has 8D state (gripper as 1D mean), sim has 9D (gripper as 2D)
    # Match training format: mean of gripper qpos
    gripper = np.mean(obs["robot0_gripper_qpos"], keepdims=True)
    proprio = np.concatenate([
        obs["robot0_eef_pos"],       # (3,)
        obs["robot0_eef_quat"],      # (4,)
        gripper,                     # (1,) — mean of 2 finger positions
    ])

    return {
        "images": images,
        "instruction": [instruction],
        "proprio": torch.from_numpy(proprio).float().unsqueeze(0),
    }


def record_video(frames: list[np.ndarray], path: str, fps: int = 20):
    """Save frames as MP4 video using av (pyav).

    Each frame can be a single image or a side-by-side composite.
    """
    import av

    h, w = frames[0].shape[:2]
    container = av.open(path, mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"

    for frame_np in frames:
        frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()
    logger.info(f"Video saved to {path} ({len(frames)} frames)")


def make_composite_frame(obs: dict) -> np.ndarray:
    """Create side-by-side frame from agentview + wrist camera."""
    agent = obs["agentview_image"]
    wrist = obs["robot0_eye_in_hand_image"]
    # Flip agentview vertically for natural viewing angle
    agent = agent[::-1].copy()
    return np.concatenate([agent, wrist], axis=1)


def evaluate_libero(
    policy,
    suite_name: str,
    camera_keys: list[str],
    n_episodes: int = 10,
    max_steps: int = 300,
    seed: int = 42,
    record_dir: str | None = None,
) -> dict:
    """Run LIBERO evaluation and return results."""
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    benchmark = get_benchmark(suite_name)()
    num_tasks = benchmark.get_num_tasks()
    logger.info(f"Suite: {suite_name}, {num_tasks} tasks, {n_episodes} episodes each")

    if record_dir:
        os.makedirs(record_dir, exist_ok=True)

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
            camera_heights=256,
            camera_widths=256,
        )
        successes = 0

        for episode in range(n_episodes):
            env.reset()
            # Reset temporal ensemble buffer for ACTHead
            if hasattr(policy.action_head, "reset_ensemble"):
                policy.action_head.reset_ensemble()
            obs = env.set_init_state(init_states[episode])

            for _ in range(5):
                obs, _, _, _ = env.step(np.zeros(7))

            frames = []

            for step in range(max_steps):
                # Record composite frame (agentview + wrist side-by-side)
                if record_dir:
                    frames.append(make_composite_frame(obs))

                batch = convert_obs_to_batch(obs, task.language, camera_keys)
                with torch.no_grad():
                    action = policy.predict(batch)
                action_np = action[0, 0].cpu().numpy()
                obs, reward, done, info = env.step(action_np)

                if info.get("success", False):
                    if record_dir:
                        frames.append(make_composite_frame(obs))
                    break

            success = int(info.get("success", False))
            successes += success
            logger.info(f"  Episode {episode}: {'SUCCESS' if success else 'FAIL'} (step {step})")

            # Save video for this episode
            if record_dir and frames:
                status = "success" if success else "fail"
                video_path = os.path.join(
                    record_dir, f"task{task_idx}_ep{episode}_{status}.mp4"
                )
                record_video(frames, video_path)

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
    parser.add_argument("--record", type=str, default=None, help="Directory to save episode videos")
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
        record_dir=args.record,
    )

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
