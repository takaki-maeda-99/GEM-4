# LIBERO Evaluation 設計仕様書

## 概要

VLA-Gemma4 の学習品質を LIBERO ベンチマーク（libero_spatial スイート）で評価する。LIBERO デモデータでファインチューニング後、シミュレーション環境で 10タスク × 10エピソード = 100エピソードのタスク成功率を計測する。

## ゴール

VLA-Gemma4 が「とりあえず動く」ことの確認。libero_spatial スイートで成功率が計測できればよい。

## 全体フロー

```
[Phase 1: 学習]
LIBERO デモデータ (HuggingFace Hub or hdf5変換)
  → VLADataset → VLAPolicy 学習 (LoRA + 4bit)
  - libero_spatial: 10タスク × 50デモ = 500デモ
  - アクション: 7DOF (EEF delta 6D + gripper 1D)
  - 観測: agentview画像 + hand画像 + state 9D
  - 言語指示: タスク名テキスト

[Phase 2: 評価]
LIBERO OffScreenRenderEnv
  → 観測取得 → VLAPolicy.predict() → env.step()
  → 10タスク × 10エピソード = 100エピソード
  → タスクごと・全体の成功率をレポート
```

## データマッピング

| LIBERO | VLAPolicy | 次元 |
|--------|-----------|------|
| `agentview_image` (128×128) | `images[0]` | [3, 128, 128] |
| `robot0_eye_in_hand_image` (128×128) | `images[1]` | [3, 128, 128] |
| `robot0_eef_pos`(3) + `robot0_eef_quat`(4) + `robot0_gripper_qpos`(2) | `proprio` | 9D |
| タスク名テキスト | `instruction` | str |
| EEF delta (6) + gripper (1) | `actions` | 7D |

## アクション正規化

初期実装ではアクション正規化なし (`normalizer_path: null`)。LIBERO の EEF delta アクションは translation と rotation でスケールが異なるが、まずは raw アクションで動作確認する。学習が収束しない場合はデータセットから mean/std を計算して Normalizer を適用する。

## 設定ファイル

学習用と評価用で設定ファイルは共通 (`configs/libero_spatial.yaml`)。ただし `cameras` フィールドは**評価スクリプトでのみ使用**される（LIBERO 観測のキー名）。学習時は LeRobot データセットのカメラキー（`observation.images.agentview` 等）が自動的に使われる。

```yaml
model_name: "google/gemma-4-E2B-it"
num_action_tokens: 1
visual_token_budget: 140

action_dim: 7
chunk_size: 1
proprio_dim: 9
proprio_key: "observation.state"

# 評価時の LIBERO 観測キー名
cameras:
  - "agentview_image"
  - "robot0_eye_in_hand_image"

# 学習時の LeRobot カメラキー名
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

## 評価スクリプト

`scripts/eval_libero.py`:

### 入力
- `--config`: LIBERO用YAML設定ファイル
- `--checkpoint`: 学習済みチェックポイント
- `--suite`: LIBEROスイート名（デフォルト: `libero_spatial`）
- `--n_episodes`: タスクあたりのエピソード数（デフォルト: 10）
- `--max_steps`: エピソードあたりの最大ステップ数（デフォルト: 300）
- `--seed`: 乱数シード（デフォルト: 42、再現性のため）
- `--output`: 結果JSON出力パス

### LIBERO Benchmark API の使い方

```python
from libero.libero.benchmark import get_benchmark

benchmark = get_benchmark(suite_name)()  # e.g. "libero_spatial"
num_tasks = benchmark.get_num_tasks()

# タスク情報の取得
task = benchmark.get_task(task_idx)
task_name = task.name
task_language = task.language  # 言語指示テキスト
bddl_path = benchmark.get_task_bddl_file_path(task_idx)

# 初期状態の取得
init_states = benchmark.get_task_init_states(task_idx)  # list of arrays
init_state = init_states[episode_idx]
```

### 処理フロー

```python
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv

benchmark = get_benchmark(suite_name)()
num_tasks = benchmark.get_num_tasks()

for task_idx in range(num_tasks):  # 10タスク
    bddl_path = benchmark.get_task_bddl_file_path(task_idx)
    task = benchmark.get_task(task_idx)
    init_states = benchmark.get_task_init_states(task_idx)

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=128,
        camera_widths=128,
    )
    successes = 0

    for episode in range(n_episodes):  # 10エピソード
        env.reset()
        # LIBERO提供の初期状態を設定（再現性のため）
        obs = env.set_init_state(init_states[episode])

        # 物理シミュレーションの安定化（5ステップのゼロアクション）
        for _ in range(5):
            obs, _, _, _ = env.step(np.zeros(7))

        for step in range(max_steps):  # 最大300ステップ
            batch = convert_obs_to_batch(obs, task.language)
            action = policy.predict(batch)  # [1, 1, 7]
            obs, reward, done, info = env.step(action[0, 0].cpu().numpy())

            if info.get("success", False):
                break

        successes += int(info.get("success", False))

    env.close()  # MuJoCo コンテキスト解放
    results[task.name] = successes / n_episodes

overall_success_rate = sum(results.values()) / len(results)
```

### 観測変換 (`convert_obs_to_batch`)

```python
def convert_obs_to_batch(obs, instruction):
    """LIBERO の obs dict → VLAPolicy の batch dict"""
    # 画像: uint8 [H,W,C] → float32 [C,H,W], batch dim追加
    agentview = torch.from_numpy(obs["agentview_image"]).permute(2,0,1).float() / 255.0
    hand = torch.from_numpy(obs["robot0_eye_in_hand_image"]).permute(2,0,1).float() / 255.0

    # Proprio: EEF pos(3) + quat(4) + gripper qpos(2) = 9D
    proprio = np.concatenate([
        obs["robot0_eef_pos"],       # (3,)
        obs["robot0_eef_quat"],      # (4,)
        obs["robot0_gripper_qpos"],  # (2,)
    ])

    return {
        "images": [agentview.unsqueeze(0), hand.unsqueeze(0)],
        "instruction": [instruction],
        "proprio": torch.from_numpy(proprio).float().unsqueeze(0),
    }
```

注: テンソルは CPU で作成される。VLAPolicy.predict() 内部で GPU に転送される。

### 出力

```json
{
  "suite": "libero_spatial",
  "n_episodes_per_task": 10,
  "max_steps": 300,
  "seed": 42,
  "tasks": {
    "pick_up_the_black_bowl_between_...": 0.3,
    "pick_up_the_black_bowl_next_to_...": 0.2,
    ...
  },
  "overall_success_rate": 0.15
}
```

## 新規ファイル

- `scripts/eval_libero.py` — LIBERO評価スクリプト
- `configs/libero_spatial.yaml` — LIBERO学習・評価用設定

## 既存ファイルの変更

なし（VLAPolicy はそのまま使える）

## 依存パッケージ

- `libero` (lerobot-libero fork: `git+https://github.com/huggingface/lerobot-libero.git`)
- `robosuite`
- `mujoco`
- `gymnasium`

## 初期実装スコープ

- [x] libero_spatial スイートのみ
- [x] 10タスク × 10エピソード
- [x] 自前の推論ループ（LeRobot Policy統合なし）
- [x] アクション正規化なし（初期実装）
- [ ] 他のスイート（libero_object, libero_goal, libero_10）は将来対応
- [ ] アクション正規化の追加（必要に応じて）
