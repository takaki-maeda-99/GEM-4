"""
eval_libero_gemma4.py

Phase 2c: VLAAdapterGemma4 の LIBERO eval pipeline (library + CLI).

Plan v2 §5 Phase 2c / Prompt v3 Part 1:
  - 原 `VLA-Adapter/experiments/robot/libero/run_libero_eval.py` の構造を参考に、Gemma 4 VLA 用の差分実装
  - Stage 1 で未検証の 3 経路を初実行:
    (1) libero simulator init (MuJoCo)
    (2) Checkpoint load (random-init or Phase 2b checkpoint)
    (3) action sampling 推論経路 (model.forward(actions=None))
    (4) BOUNDS_Q99 denormalize (dataset_statistics.json + mask 適用)
    (5) Simulator step + next obs

CLI usage (Phase 2g 本番 eval):
  CUDA_VISIBLE_DEVICES=0 .venv-gemma4/bin/python scripts/gemma4/eval_libero_gemma4.py \\
      --checkpoint_path runs/gemma4/.../latest_checkpoint.pt \\
      --task_suite_name libero_spatial --num_trials_per_task 10

Library usage (Phase 2c dry-run、Phase 2f 10k sanity、Phase 2i bidirectional eval 他で import):
  from eval_libero_gemma4 import EvalConfig, load_action_proprio_stats, evaluate

BOUNDS_Q99 denormalize (train 側 inverse):
  train: mask=True dim は   x_norm = clip(2*(x-q01)/(q99-q01) - 1, -1, 1)
         mask=False dim は  x_norm = x (unchanged、LIBERO gripper が [0,1] のまま)
  eval:  mask=True dim は   x = (x_norm + 1) / 2 * (q99 - q01) + q01
         mask=False dim は  x = x_norm (unchanged)

Gripper action post-process (LIBERO env 向け):
  (a) normalize_gripper_action(binarize=True): [0,1] → {-1, +1}
  (b) invert_gripper_action: sign flip (RLDS 0=close, 1=open → LIBERO -1=open, +1=close)
"""
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# LIBERO MuJoCo offscreen render (README:140 の `AttributeError: eglQueryString` 対策、egl 優先)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import draccus
import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer

# LIBERO の benchmark/__init__.py:164 が `torch.load(init_states_path)` を
# weights_only=True (PyTorch 2.6+ default) で呼び、numpy._reconstruct が reject される issue の workaround。
# 我々の LIBERO は local clone で pip -e install、信頼できる source のため weights_only=False で問題なし。
_ORIG_TORCH_LOAD = torch.load
def _torch_load_weights_only_false(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _ORIG_TORCH_LOAD(*args, **kwargs)
torch.load = _torch_load_weights_only_false

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VLA_ROOT = REPO_ROOT / "VLA-Adapter"
SCRIPTS_GEMMA4 = Path(__file__).resolve().parent
VLA_SCRIPTS = VLA_ROOT / "vla-scripts"
EXPERIMENTS = VLA_ROOT / "experiments"
sys.path.insert(0, str(VLA_ROOT))          # `prismatic.*` と `experiments.robot.*` import 用
sys.path.insert(0, str(SCRIPTS_GEMMA4))    # `test_08_data_pipeline` import 用
sys.path.insert(0, str(VLA_SCRIPTS))       # `finetune_gemma4` import 用

# ------- TF lanczos3 resize は optional (openvla_utils.resize_image_for_policy と同等) -------
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")    # CUDA context 干渉回避 (R16 趣旨、single-GPU でも保護)

# ------- Gemma 4 側資産 -------
from test_08_data_pipeline import PROMPT_MAX_LEN  # noqa: E402
from prismatic.vla.constants_gemma4 import (  # noqa: E402
    ACTION_TOKEN_BEGIN_IDX,
    NUM_ACTION_TOKENS,
    NUM_VISION_TOKENS,
    PROPRIO_PLACEHOLDER_IDX,
    VISION_PLACEHOLDER_BEGIN_IDX,
)
from finetune_gemma4 import build_model  # noqa: E402

# ------- LIBERO 側資産 -------
# NOTE: `experiments.robot.libero.libero_utils` / `experiments.robot.robot_utils` は
# 上流で `openvla_utils.py` に連鎖 import され、それが transformers 5.5.4 で削除済の
# `AutoModelForVision2Seq` に依存して import 失敗する。
# 解決策として必要な helper を eval script に inline (原実装 libero_utils.py / robot_utils.py から verbatim、
# OpenVLA 以外には普遍的な pure function なので copy でも機能等価)。
import math  # noqa: E402
import random  # noqa: E402

from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402


def set_seed_everywhere(seed: int) -> None:
    """from experiments/robot/robot_utils.py"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """from experiments/robot/robot_utils.py: [0,1] → [-1,+1] → sign"""
    out = action.copy()
    orig_low, orig_high = 0.0, 1.0
    out[..., -1] = 2.0 * (out[..., -1] - orig_low) / (orig_high - orig_low) - 1.0
    if binarize:
        out[..., -1] = np.sign(out[..., -1])
    return out


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """from experiments/robot/robot_utils.py: sign flip (-1=open, +1=close for LIBERO)"""
    out = action.copy()
    out[..., -1] *= -1.0
    return out


def get_libero_env(task, model_family: str, resolution: int = 256):
    """from experiments/robot/libero/libero_utils.py"""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """from experiments/robot/libero/libero_utils.py"""
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(obs):
    """from experiments/robot/libero/libero_utils.py — 180度回転で train 前処理に一致"""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]
    return img


def get_libero_wrist_image(obs):
    """from experiments/robot/libero/libero_utils.py"""
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]
    return img


def quat2axisangle(quat):
    """from experiments/robot/libero/libero_utils.py (robosuite 準拠)"""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

# ======================================================================
# Constants (LIBERO env 仕様)
# ======================================================================
TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
POLICY_IMAGE_SIZE = 224
ENV_IMAGE_SIZE = 256
NUM_ACTIONS_CHUNK = 8
ACTION_DIM = 7
PROPRIO_DIM = 8


# ======================================================================
# Config
# ======================================================================
@dataclass
class EvalConfig:
    # fmt: off
    # --- Model ---
    gemma_model_id: str = "google/gemma-4-E2B"
    vision_backbone_id: str = "dinosiglip-vit-so-224px"  # DEPRECATED (rev 3)、vision_backbone_type 参照
    vision_backbone_type: str = "gemma4_native"         # "gemma4_native" | "dinosiglip" | "siglip"
    siglip_use_tensor_transform: bool = True            # SigLIP で no-PIL GPU transform 使用
    training_mode: str = "quality"                       # "quality" | "speed"
    use_xvla_style: bool = False                         # 2026-04-24 #014: X-VLA 流 action_head
    use_wrist_bridge: bool = False                       # 2026-04-24 #015: wrist per-layer Bridge cross-attn
    use_proper_ffn: bool = False                         # 2026-04-24 #016: proper transformer FFN
    feature_norm_type: str = "identity"                  # 2026-04-24 #018: identity | layer_norm
    wrist_bridge_layer_mode: str = "per_layer"           # 2026-04-24 #019: per_layer | final_broadcast
    num_action_head_blocks: int = 24                     # 2026-04-25 #021: action_head の block 数 (24=paper、35=Gemma4 全層)
    # LoRA config (Mode A ckpt load 時必須、ckpt の LoRA shape と一致させる)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    checkpoint_path: str = ""          # empty = random-init (dry-run 用)

    # --- Data / denormalize stats ---
    dataset_statistics_path: str = "VLA-Adapter/outputs/LIBERO-Spatial-Pro/dataset_statistics.json"
    unnorm_key: str = "libero_spatial_no_noops"

    # --- LIBERO env ---
    task_suite_name: str = "libero_spatial"
    task_id: int = 0                   # dry-run は 0、full eval は全 task iterate
    num_trials_per_task: int = 1       # dry-run は 1、full eval は 10-50
    num_tasks_limit: int = 0           # 0 = 全 task (dry-run は 1 に制限)
    num_steps_wait: int = 10
    num_open_loop_steps: int = 8
    env_img_res: int = ENV_IMAGE_SIZE

    # --- Output ---
    output_dir: str = "runs/gemma4/eval"
    run_id_note: str = ""
    save_video: bool = True
    seed: int = 7
    # fmt: on


# ======================================================================
# Denormalize utilities (BOUNDS_Q99)
# ======================================================================
def load_action_proprio_stats(dataset_statistics_path: Path, unnorm_key: str) -> Tuple[dict, dict]:
    """dataset_statistics.json から action / proprio の q01/q99/mask を取り出す."""
    all_stats = json.loads(Path(dataset_statistics_path).read_text())
    assert unnorm_key in all_stats, \
        f"unnorm_key '{unnorm_key}' not in {dataset_statistics_path} (keys: {list(all_stats.keys())})"
    per_dataset = all_stats[unnorm_key]
    action = per_dataset["action"]
    proprio = per_dataset["proprio"]
    # action に mask が入っている、proprio は無いので全 True
    action_stats = {
        "q01": np.array(action["q01"], dtype=np.float32),
        "q99": np.array(action["q99"], dtype=np.float32),
        "mask": np.array(action.get("mask", [True] * len(action["q01"])), dtype=bool),
    }
    proprio_stats = {
        "q01": np.array(proprio["q01"], dtype=np.float32),
        "q99": np.array(proprio["q99"], dtype=np.float32),
        "mask": np.array(proprio.get("mask", [True] * len(proprio["q01"])), dtype=bool),
    }
    return action_stats, proprio_stats


def normalize_proprio_bounds_q99(proprio_raw: np.ndarray, proprio_stats: dict) -> np.ndarray:
    """BOUNDS_Q99 の forward (train 時と同じ)、mask=True dim のみ適用."""
    q01 = proprio_stats["q01"]
    q99 = proprio_stats["q99"]
    mask = proprio_stats["mask"]
    out = proprio_raw.astype(np.float32).copy()
    # mask=True dim のみ normalize、それ以外は raw のまま
    # clip to [-1, 1]
    denom = (q99 - q01) + 1e-8
    norm = np.clip(2.0 * (proprio_raw - q01) / denom - 1.0, -1.0, 1.0)
    out[mask] = norm[mask]
    return out


def denormalize_action_bounds_q99(action_norm: np.ndarray, action_stats: dict) -> np.ndarray:
    """BOUNDS_Q99 の inverse、mask=True dim のみ適用 (gripper は unchanged)."""
    q01 = action_stats["q01"]
    q99 = action_stats["q99"]
    mask = action_stats["mask"]
    out = action_norm.astype(np.float32).copy()
    # 各 dim ごとに適用 (broadcast)、mask=False は unchanged
    action_un = (action_norm + 1.0) / 2.0 * (q99 - q01) + q01
    # broadcast を ... 全 leading dim で適用するため、mask を最終 dim で index
    for i in range(len(mask)):
        if mask[i]:
            out[..., i] = action_un[..., i]
    return out


# ======================================================================
# Image resize (train-matching via TF lanczos3、openvla_utils.resize_image_for_policy 準拠)
# ======================================================================
def resize_image_for_policy(img: np.ndarray, resize_size: int = POLICY_IMAGE_SIZE) -> np.ndarray:
    """train 時の RLDS frame_transform (lanczos3) と同じ resize を eval でも適用."""
    img_tf = tf.image.encode_jpeg(img)
    img_tf = tf.io.decode_image(img_tf, expand_animations=False, dtype=tf.uint8)
    img_tf = tf.image.resize(img_tf, (resize_size, resize_size), method="lanczos3", antialias=True)
    img_tf = tf.cast(tf.clip_by_value(tf.round(img_tf), 0, 255), tf.uint8)
    return img_tf.numpy()


# ======================================================================
# Input construction (Gemma4BatchTransform の inference 版、B=1 固定)
# ======================================================================
def build_input_ids(language: str, tokenizer: Any, prompt_max_len: int = PROMPT_MAX_LEN) -> torch.Tensor:
    """Gemma4BatchTransform の tokenize + placeholder 構築を 1 sample 分で再現."""
    text = f"What action should the robot take to {language.lower().strip()}?"
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) > prompt_max_len:
        ids = ids[:prompt_max_len]
    else:
        pad = [tokenizer.pad_token_id] * (prompt_max_len - len(ids))
        ids = ids + pad
    full = (
        [tokenizer.bos_token_id]
        + list(range(VISION_PLACEHOLDER_BEGIN_IDX, VISION_PLACEHOLDER_BEGIN_IDX + NUM_VISION_TOKENS))
        + ids
        + [PROPRIO_PLACEHOLDER_IDX]
        + list(range(ACTION_TOKEN_BEGIN_IDX, ACTION_TOKEN_BEGIN_IDX + NUM_ACTION_TOKENS))
        + [tokenizer.eos_token_id]
    )
    return torch.tensor(full, dtype=torch.long)


def get_action_gemma4(
    model_vla,
    tokenizer,
    obs_for_model: Dict[str, np.ndarray],
    language_instruction: str,
    proprio_stats: dict,
    action_stats: dict,
    device: torch.device,
) -> np.ndarray:
    """single obs で forward 実行、denormalized action chunk (NUM_ACTIONS_CHUNK, ACTION_DIM) を返す.

    rev 3 (Task 6-8): pixel_values は {"scene", "wrist"} dict schema。
    - scene: (B, 3, H, W) float [0, 255] CPU、encode_scene 内部で transform
    - wrist: (B, 3, 224, 224) bf16 on GPU、WristResNet18 入力 (ImageNet 正規化相当で OK)
    """
    # --- Images (new {scene, wrist} schema、rev 3 Task 6-8) ---
    # obs_for_model["full_image"], ["wrist_image"] は (H, W, 3) uint8 numpy
    scene_np = obs_for_model["full_image"]
    wrist_np = obs_for_model["wrist_image"]
    # scene: raw float [0, 255] CPU、encode_scene が内部で resize + normalize + backbone
    scene_t = torch.from_numpy(scene_np).permute(2, 0, 1).float().unsqueeze(0)  # (1, 3, H, W) CPU
    # wrist: 224×224 resize + ImageNet 正規化 + bf16 on GPU (WristResNet18 convention)
    import torch.nn.functional as F
    w = torch.from_numpy(wrist_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0  # (1, 3, H, W) [0,1]
    w = F.interpolate(w, size=(224, 224), mode="bilinear", antialias=True)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    w = (w - mean) / std
    wrist_t = w.to(device, dtype=torch.bfloat16)
    pv = {
        "scene": scene_t,       # model.encode_scene が内部で handle
        "wrist": wrist_t,
    }
    # --- input_ids ---
    input_ids = build_input_ids(language_instruction, tokenizer).unsqueeze(0).to(device)
    # --- Proprio (normalize に train と整合) ---
    state_norm = normalize_proprio_bounds_q99(obs_for_model["state"], proprio_stats)
    proprio = torch.tensor(state_norm, dtype=torch.bfloat16).unsqueeze(0).to(device)
    # --- Forward ---
    was_training = model_vla.training
    model_vla.eval()
    with torch.no_grad():
        predicted = model_vla(pv, input_ids, proprio, actions=None)
    if was_training:
        model_vla.train()
    # predicted: (B=1, NUM_ACTIONS_CHUNK, ACTION_DIM) bf16 on GPU
    action_norm = predicted[0].detach().float().cpu().numpy()  # (8, 7)
    # --- Denormalize (mask=False gripper 以外を q01/q99 に戻す) ---
    action = denormalize_action_bounds_q99(action_norm, action_stats)
    return action  # (8, 7) denormalized (gripper 未処理)


# ======================================================================
# Checkpoint load (trainable state のみ、optimizer/scheduler 不要)
# ======================================================================
def load_model_state(model_vla, checkpoint_path: Optional[Path], device: torch.device) -> dict:
    if not checkpoint_path:
        print("[eval] checkpoint_path is empty → using RANDOM-INIT model (Phase 2c dry-run 想定)")
        return {"loaded": False, "checkpoint_path": None}
    checkpoint_path = Path(checkpoint_path).resolve()
    assert checkpoint_path.exists(), f"checkpoint not found: {checkpoint_path}"
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = payload["trainable_state_dict"]
    missing, unexpected = model_vla.load_state_dict(state, strict=False)
    # non-frozen keys must be in ckpt: action_head, proprio/vision_projector, wrist_encoder,
    # action_queries, soft_prompt_library, AND Mode A の lora_* weights (llm.* 配下に在る)
    # 2026-04-24 #015/#016: k_wrist/v_wrist/gating_factor_wrist/ffn_up/ffn_down/norm1/norm2/wrist_projector_bridge
    # は use_wrist_bridge / use_proper_ffn 有効時のみ使用。pre-#015/#016 ckpt には無いので、
    # 該当 flag で model 構築中に追加された random-init param が missing になっても許容 (実行時 branch で skip される)。
    _NEW_SINCE_15_16 = (
        "k_wrist.", "v_wrist.", "gating_factor_wrist",
        "wrist_projector_bridge",
        "ffn_up.", "ffn_down.",
        ".norm1.", ".norm2.",
        "feature_norm.",   # #018: LayerNorm variant (Identity の旧 ckpt には無い)
    )
    relevant_missing = [
        k for k in missing
        if not k.startswith("vision_backbone.")
        and (not k.startswith("llm.") or "lora_" in k)
        and not any(t in k for t in _NEW_SINCE_15_16)
    ]
    assert not relevant_missing, f"missing (non-frozen) keys: {relevant_missing[:5]}"
    # unexpected keys (soft_prompt_library 等、LIBERO eval では num_pretrain_datasets=0 で model が持たない) は warn のみ。
    # action head への flow では LIBERO 用の h_sp=None 運用で無問題。
    if unexpected:
        print(f"[eval] WARN ignoring unexpected ckpt keys (e.g., soft_prompt_library): {unexpected[:3]} total={len(unexpected)}")
    print(f"[eval] loaded checkpoint: {checkpoint_path}")
    return {
        "loaded": True,
        "checkpoint_path": str(checkpoint_path),
        "gradient_step_idx_at_save": payload.get("gradient_step_idx"),
        "current_lr_at_save": payload.get("current_lr"),
    }


# ======================================================================
# Episode runner (dry-run + full eval で共用)
# ======================================================================
def run_episode(
    model_vla,
    tokenizer,
    env,
    task_description: str,
    action_stats: dict,
    proprio_stats: dict,
    max_steps: int,
    cfg: EvalConfig,
    initial_state,
    device: torch.device,
) -> dict:
    """1 episode 実行、success + metrics + replay_images + action trajectory を返す."""
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    action_queue: deque = deque(maxlen=cfg.num_open_loop_steps)
    t = 0
    replay_images = []
    action_history_denorm = []    # denormalize 後 (env 直前の値)
    action_history_normalized = []  # model 出力 (normalize 空間)
    proprio_history = []           # raw proprio
    model_query_times = []
    success = False
    error_str = None

    try:
        while t < max_steps + cfg.num_steps_wait:
            # Settling phase (objects stabilize)
            if t < cfg.num_steps_wait:
                obs, _, _, _ = env.step(get_libero_dummy_action("gemma4"))
                t += 1
                continue

            # --- Prepare obs for model ---
            img_raw = get_libero_image(obs)        # (256, 256, 3)
            wrist_raw = get_libero_wrist_image(obs)
            img_resized = resize_image_for_policy(img_raw, POLICY_IMAGE_SIZE)
            wrist_resized = resize_image_for_policy(wrist_raw, POLICY_IMAGE_SIZE)
            state = np.concatenate((
                obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            ))
            assert state.shape == (PROPRIO_DIM,), f"proprio shape {state.shape} != ({PROPRIO_DIM},)"
            proprio_history.append(state.copy())

            model_obs = {
                "full_image": img_resized,
                "wrist_image": wrist_resized,
                "state": state,
            }
            replay_images.append(img_raw)

            # --- Query model if queue empty ---
            if len(action_queue) == 0:
                t_query = time.time()
                actions_denorm = get_action_gemma4(
                    model_vla=model_vla,
                    tokenizer=tokenizer,
                    obs_for_model=model_obs,
                    language_instruction=task_description,
                    proprio_stats=proprio_stats,
                    action_stats=action_stats,
                    device=device,
                )
                model_query_times.append(time.time() - t_query)
                assert actions_denorm.shape == (NUM_ACTIONS_CHUNK, ACTION_DIM), \
                    f"actions shape {actions_denorm.shape} != ({NUM_ACTIONS_CHUNK}, {ACTION_DIM})"
                action_queue.extend([actions_denorm[i] for i in range(NUM_ACTIONS_CHUNK)])

            # --- Pop + gripper post-process + env step ---
            action = action_queue.popleft()      # (7,) denormalized、gripper は [0,1]
            action_history_denorm.append(action.copy())

            # (a) [0,1] → {-1, +1}
            action_proc = normalize_gripper_action(action, binarize=True)
            # (b) sign flip (LIBERO: -1=open, +1=close)
            action_proc = invert_gripper_action(action_proc)
            obs, reward, done, info = env.step(action_proc.tolist())
            if done:
                success = True
                break
            t += 1
    except Exception as e:
        error_str = f"{type(e).__name__}: {e}"
        print(f"[eval] Episode exception: {error_str}")

    # --- Stats ---
    action_hist = np.array(action_history_denorm) if action_history_denorm else np.zeros((0, ACTION_DIM))
    proprio_hist = np.array(proprio_history) if proprio_history else np.zeros((0, PROPRIO_DIM))
    return {
        "success": success,
        "error": error_str,
        "num_env_steps": t,
        "num_model_queries": len(model_query_times),
        "replay_images": replay_images,
        "action_history_denorm": action_hist,
        "proprio_history": proprio_hist,
        "model_query_time_median_s": float(np.median(model_query_times)) if model_query_times else 0.0,
        "model_query_time_total_s": float(np.sum(model_query_times)),
    }


# ======================================================================
# Top-level evaluation (CLI + library entrypoint)
# ======================================================================
def evaluate(cfg: EvalConfig) -> dict:
    """タスク × エピソードを iterate、success rate + metrics を返す."""
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    set_seed_everywhere(cfg.seed)

    output_dir = Path(cfg.output_dir)
    if cfg.run_id_note:
        output_dir = output_dir / f"{cfg.task_suite_name}--{cfg.run_id_note}"
    else:
        output_dir = output_dir / f"{cfg.task_suite_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] cfg: {cfg}")
    print(f"[eval] output_dir: {output_dir.resolve()}")

    # --- Stats load ---
    stats_path = Path(cfg.dataset_statistics_path)
    if not stats_path.is_absolute():
        stats_path = REPO_ROOT / stats_path
    action_stats, proprio_stats = load_action_proprio_stats(stats_path, cfg.unnorm_key)
    print(f"[eval] action q01/q99 range: [{action_stats['q01'].min():.3f}, {action_stats['q99'].max():.3f}]  mask={action_stats['mask']}")
    print(f"[eval] proprio q01/q99 range: [{proprio_stats['q01'].min():.3f}, {proprio_stats['q99'].max():.3f}]")

    # --- Model build ---
    # finetune_gemma4.build_model は FinetuneConfig を要求するため、minimal shim を渡す
    print("[eval] Building model...")
    t0 = time.time()
    from finetune_gemma4 import FinetuneConfig
    # Rev 3 (Task 6+): vision_backbone_type で gemma4_native / dinosiglip / siglip 切替、cfg から渡す
    model_cfg = FinetuneConfig(
        gemma_model_id=cfg.gemma_model_id,
        vision_backbone_type=cfg.vision_backbone_type,
        siglip_use_tensor_transform=cfg.siglip_use_tensor_transform,
        proprio_dim=PROPRIO_DIM, action_dim=ACTION_DIM, num_action_chunks=NUM_ACTIONS_CHUNK,
        training_mode=cfg.training_mode,
        use_xvla_style=cfg.use_xvla_style,
        use_wrist_bridge=cfg.use_wrist_bridge,
        use_proper_ffn=cfg.use_proper_ffn,
        feature_norm_type=cfg.feature_norm_type,
        wrist_bridge_layer_mode=cfg.wrist_bridge_layer_mode,
        num_action_head_blocks=cfg.num_action_head_blocks,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_target_modules=cfg.lora_target_modules,
    )
    model_vla, tok = build_model(model_cfg, device)
    print(f"[eval] model built in {time.time()-t0:.1f}s")
    model_vla.eval()

    # --- Checkpoint load ---
    load_info = load_model_state(model_vla, cfg.checkpoint_path, device)

    # --- Benchmark init ---
    print("[eval] Initializing LIBERO benchmark...")
    t0 = time.time()
    benchmark_dict = benchmark.get_benchmark_dict()
    assert cfg.task_suite_name in benchmark_dict, \
        f"unknown task suite: {cfg.task_suite_name} (available: {list(benchmark_dict.keys())})"
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks
    print(f"[eval] task suite '{cfg.task_suite_name}' has {num_tasks} tasks (init took {time.time()-t0:.1f}s)")

    # dry-run / limited-task mode
    if cfg.num_tasks_limit > 0:
        task_ids = list(range(min(cfg.num_tasks_limit, num_tasks)))
    else:
        task_ids = list(range(num_tasks))

    max_steps = TASK_MAX_STEPS.get(cfg.task_suite_name, 220)

    # --- Run ---
    aggregate: List[Dict[str, Any]] = []
    for task_id in task_ids:
        if cfg.num_tasks_limit > 0 and task_id != cfg.task_id and len(task_ids) == 1:
            # dry-run 用 single task override
            task_id = cfg.task_id
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, "gemma4", resolution=cfg.env_img_res)
        print(f"\n=== Task {task_id}: {task_description} ===")

        task_successes = 0
        for ep_idx in range(cfg.num_trials_per_task):
            init_state = initial_states[ep_idx] if ep_idx < len(initial_states) else None
            print(f"  [episode {ep_idx+1}/{cfg.num_trials_per_task}] initial_state: {type(init_state).__name__}")
            t0 = time.time()
            episode = run_episode(
                model_vla=model_vla,
                tokenizer=tok,
                env=env,
                task_description=task_description,
                action_stats=action_stats,
                proprio_stats=proprio_stats,
                max_steps=max_steps,
                cfg=cfg,
                initial_state=init_state,
                device=device,
            )
            ep_sec = time.time() - t0
            task_successes += int(episode["success"])

            # --- Summary ---
            ah = episode["action_history_denorm"]
            ph = episode["proprio_history"]
            ep_summary = {
                "task_id": task_id,
                "task_description": task_description,
                "episode_idx": ep_idx,
                "success": episode["success"],
                "error": episode["error"],
                "num_env_steps": episode["num_env_steps"],
                "num_model_queries": episode["num_model_queries"],
                "model_query_median_s": episode["model_query_time_median_s"],
                "model_query_total_s": episode["model_query_time_total_s"],
                "episode_wall_sec": round(ep_sec, 2),
                "action_denorm_range_per_dim": (
                    [[float(ah[:, d].min()), float(ah[:, d].max())] for d in range(ACTION_DIM)]
                    if ah.size else []
                ),
                "action_denorm_mean_per_dim": (
                    [float(ah[:, d].mean()) for d in range(ACTION_DIM)] if ah.size else []
                ),
                "action_denorm_std_per_dim": (
                    [float(ah[:, d].std()) for d in range(ACTION_DIM)] if ah.size else []
                ),
                "proprio_range_per_dim": (
                    [[float(ph[:, d].min()), float(ph[:, d].max())] for d in range(PROPRIO_DIM)]
                    if ph.size else []
                ),
                "replay_n_frames": len(episode["replay_images"]),
            }
            aggregate.append(ep_summary)
            print(f"  [episode {ep_idx+1}] success={episode['success']}  steps={episode['num_env_steps']}  queries={episode['num_model_queries']}  wall={ep_sec:.1f}s")
            if episode["error"]:
                print(f"  [episode {ep_idx+1}] ERROR: {episode['error']}")

            # --- Save video (optional) ---
            if cfg.save_video and episode["replay_images"]:
                try:
                    import imageio
                    vid_path = output_dir / f"task{task_id}--ep{ep_idx}--success={episode['success']}.mp4"
                    writer = imageio.get_writer(vid_path, fps=30)
                    for im in episode["replay_images"]:
                        writer.append_data(im)
                    writer.close()
                    print(f"  [episode {ep_idx+1}] video saved → {vid_path}")
                except Exception as e:
                    print(f"  [episode {ep_idx+1}] video save failed: {e}")

        print(f"  Task {task_id} success rate: {task_successes}/{cfg.num_trials_per_task} = {task_successes/cfg.num_trials_per_task*100:.1f}%")
        env.close()
        del env

    # --- Final summary ---
    total_eps = len(aggregate)
    total_succ = sum(1 for r in aggregate if r["success"])
    overall_rate = total_succ / total_eps if total_eps > 0 else 0.0
    summary = {
        "phase": "2c" if cfg.checkpoint_path == "" else "eval",
        "cfg": {k: str(v) if isinstance(v, Path) else v for k, v in cfg.__dict__.items()},
        "checkpoint_load_info": load_info,
        "action_stats": {
            "q01": action_stats["q01"].tolist(),
            "q99": action_stats["q99"].tolist(),
            "mask": action_stats["mask"].tolist(),
        },
        "num_tasks_evaluated": len(task_ids),
        "total_episodes": total_eps,
        "total_successes": total_succ,
        "overall_success_rate": overall_rate,
        "episodes": aggregate,
    }
    out_json = output_dir / "eval_result.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n[eval] Summary JSON: {out_json}")
    print(f"[eval] Overall success rate: {total_succ}/{total_eps} = {overall_rate*100:.1f}%")
    return summary


@draccus.wrap()
def main(cfg: EvalConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
