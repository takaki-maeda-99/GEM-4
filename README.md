# VLA-Gemma4

Gemma 4 E2B をバックボーンとした Vision-Language-Action (VLA) モデル。ロボットマニピュレーションにおいて、画像・言語指示・プロプリオセプションから EEF デルタポーズアクションを予測する。

## Key Contribution: PLE-Compatible Custom Token Injection

Gemma 4 の Per-Layer Embeddings (PLE) アーキテクチャは `input_ids` を前提とした設計であり、従来の VLA のように `inputs_embeds` を直接結合するアプローチが使えない。本実装では Gemma 4 自身がビジョン/オーディオトークンを処理するのと同じ **「PAD プレースホルダー + 埋め込み上書き」パターン** を応用し、プロプリオセプション・ACT トークンの注入を実現した。

```
カメラ画像 → Gemma4 ViT → embed_vision → [画像トークン]
言語指示  → Gemma4 Tokenizer            → [言語トークン]
プロプリオ → ProprioEncoder(MLP)         → [proprioトークン]  ← PAD IDでPLE計算、埋め込み上書き
                                           [ACTトークン]      ← 同上

→ Gemma4 LLM (PLE付き) → [ACT]トークンの隠れ状態 → ActionHead → Δアクション
```

## Architecture

| Component | Description |
|-----------|-------------|
| Backbone | Gemma 4 E2B (2.3B effective, 4-bit quantization) |
| Vision | 内蔵 ViT (~150M params) + embed_vision projector |
| Action Head | 交換可能 — MLP (実装済み) / ACT / FlowMatching / UniAct |
| Proprio | MLP encoder → single token embedding |
| Feature extraction | Learnable [ACT] token の最終隠れ状態 |
| Training | LoRA (0.026% trainable) or Full FT |
| Data | LeRobot format (Open X-Embodiment + custom) |

## Results

RTX 5070 Ti (16GB VRAM) での検証:

| Metric | Value |
|--------|-------|
| VRAM | 6.80 GB peak |
| Trainable params | 1.3M / 5.1B (0.026%) |
| Loss (20 steps) | 1.70 → 0.63 |
| MSE (20 steps) | 0.44 → 0.10 |
| Speed | ~0.15 sec/step |
| Dataset | `lerobot/aloha_sim_transfer_cube_human` |
| Tests | 36/36 passed |

## Quick Start

### Install

```bash
pip install -e ".[dev]"
```

### Quick Test (動作確認)

```bash
python3 scripts/quick_test.py
```

20ステップの学習 + 3サンプルの推論を実行。デフォルトで `vla_gemma4/configs/aloha_sim_test.yaml` を使用。

### Training

```bash
python3 vla_gemma4/scripts/train.py --config vla_gemma4/configs/aloha_sim_test.yaml --output_dir outputs/
```

### Evaluation

```bash
python3 vla_gemma4/scripts/eval.py \
  --config vla_gemma4/configs/aloha_sim_test.yaml \
  --checkpoint outputs/checkpoint_step_500.pt \
  --output eval_results.json
```

### Tests

```bash
pytest tests/ -v
```

## Project Structure

```
vla_gemma4/
├── model/
│   ├── vla_policy.py          # VLAPolicy: PLE対応 encode + Gemma4 統合
│   ├── proprio_encoder.py     # ProprioEncoder: proprio → token embedding
│   └── action_heads/
│       ├── base.py             # ActionHead ABC (交換可能インターフェース)
│       └── mlp_head.py         # MLPHead: MSE + BCE loss
├── data/
│   ├── dataset.py              # VLADataset: LeRobot wrapper
│   ├── normalizer.py           # Action normalization (mean/std)
│   ├── collate.py              # Custom collate_fn for DataLoader
│   └── transforms.py           # Image augmentation
├── training/
│   └── trainer.py              # VLATrainer: HF Accelerate + AMP + LR scheduler
├── scripts/
│   ├── train.py                # Training CLI (LoRA / Full FT)
│   └── eval.py                 # Evaluation CLI (MSE / L1 / gripper accuracy)
└── configs/
    ├── base.yaml               # Default config
    └── aloha_sim_test.yaml     # Quick test config (4-bit + LoRA)
```

## Configuration

設定は YAML ファイルで管理。以下を切り替え可能:

- `model_name` — バックボーンモデル (サイズ変更)
- `cameras` — カメラリスト (数の変更)
- `action_head.type` — ヘッド種類
- `chunk_size` — 1 (1ステップ) or N (チャンキング)
- `training.strategy` — `"full"` or `"lora"`
- `inference.quantization` — `null` / `"8bit"` / `"4bit"`

## Design for Extensibility

- **アクションヘッド交換**: `ActionHead` ABC を継承して `compute_loss()` / `predict()` を実装するだけ
- **カメラ数可変**: 設定ファイルの `cameras` リストを変更
- **チャンキング**: `chunk_size` を変更 (MLPHead は両方対応)
- **バックボーン**: `model_name` を変更 (e.g. `google/gemma-4-4B-it`)
- **データセット**: LeRobot 形式であればどれでも対応

## Roadmap

- [ ] ACTHead (CVAE + chunking)
- [ ] FlowMatchingHead
- [ ] UniActHead (VQ codebook)
- [ ] Simulation evaluation (SIMPLER / ManiSkill)
- [ ] Multi-dataset training
- [ ] Observation history (temporal context)
