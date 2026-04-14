# VLA-Gemma4 設計仕様書

## 概要

Gemma 4 E2B をバックボーンとした Vision-Language-Action (VLA) モデル。ロボットマニピュレーションタスクにおいて、画像・言語指示・プロプリオセプションから EEF デルタポーズを予測する。アクションヘッドを交換可能な設計とし、MLP / ACT / Flow Matching / UniAct 等の手法を比較検証できるようにする。

## アーキテクチャ

### 方式: トークン統合型

すべての入力をトークン列として Gemma 4 の LLM バックボーンに統合する。Gemma 4 E2B 内蔵のビジョンエンコーダ (~150M params) をそのまま活用し、画像 → トークン変換は既存のパイプラインに乗せる。

```
観測（timestep t）:
  カメラ1画像 ──→ Gemma4 ViT → プロジェクション → [画像トークン1]
  カメラ2画像 ──→ Gemma4 ViT → プロジェクション → [画像トークン2]
  ...
  カメラN画像 ──→ Gemma4 ViT → プロジェクション → [画像トークンN]
  タスク指示テキスト ──→ Gemma4 Tokenizer   → [言語トークン]
  プロプリオ(7d) ──→ ProprioEncoder(MLP)    → [proprioトークン]

  [画像トークン1] ... [画像トークンN] [言語トークン] [proprioトークン] [ACTトークン]
      → Gemma 4 LLM バックボーン（End-to-End 学習）
      → [ACT]トークン位置の最終隠れ状態を特徴として抽出
      → ActionHead.predict() → Δアクション [B, T, 7]
```

### バックボーン: Gemma 4 E2B

- **総パラメータ**: 2.3B effective (5.1B with embeddings)
- **レイヤー数**: 35
- **コンテキスト長**: 128K トークン
- **ビジョンエンコーダ**: ~150M params、可変アスペクト比・解像度対応
- **ビジュアルトークンバジェット**: 70 / 140 / 280 / 560 / 1120 から選択可能
- **HuggingFace**: `AutoModelForMultimodalLM.from_pretrained("google/gemma-4-E2B-it")`
- バックボーンサイズは `model_name` の変更のみで切り替え可能（例: `google/gemma-4-4B-it`）

### アクション空間

- **形式**: EEF デルタポーズ（相対座標）
- **次元**: 7 (Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper)
- **出力形状**: `[B, T, 7]` — T=1 で1ステップ予測、T>1 でチャンキング予測

### 特徴抽出

学習可能な `[ACT]` トークン（特殊トークン埋め込み）をトークン列の末尾に追加し、LLM 最終層のその位置の隠れ状態をアクションヘッドへの入力とする。これにより固定長の特徴ベクトル `[B, D]` が得られる。

## モジュール構成

```
vla_gemma4/
├── model/
│   ├── vla_policy.py          # VLAPolicy: 全体を束ねる nn.Module
│   ├── proprio_encoder.py     # ProprioEncoder: プロプリオ → トークン埋め込み
│   └── action_heads/
│       ├── base.py             # ActionHead ABC（共通インターフェース）
│       ├── mlp_head.py         # MLPHead（回帰、初期実装）
│       ├── act_head.py         # ACTHead（CVAE + チャンキング、後日実装）
│       ├── flow_head.py        # FlowMatchingHead（後日実装）
│       └── uniact_head.py      # UniActHead（VQ コードブック、後日実装）
├── data/
│   ├── dataset.py              # LeRobot データセットのラッパー
│   └── transforms.py           # 画像前処理・データ拡張
├── training/
│   ├── trainer.py              # 学習ループ
│   └── configs/                # 設定ファイル
├── scripts/
│   ├── train.py                # 学習スクリプト
│   └── eval.py                 # 評価スクリプト
└── configs/
    ├── base.yaml               # 共通設定
    ├── mlp_head.yaml           # MLP ヘッド用
    └── act_head.yaml           # ACT ヘッド用
```

### VLAPolicy

全体を束ねるメインの `nn.Module`。

**責務:**
- Gemma 4 モデルのロードと管理
- 各入力（画像・言語・プロプリオセプション）のトークン化と結合
- `[ACT]` トークンの追加と特徴抽出
- アクションヘッドへの特徴受け渡し
- `compute_loss()` / `predict()` をアクションヘッドに委譲

### ProprioEncoder

プロプリオセプション（7次元: EEF 位置 + 回転 + グリッパー）を LLM の埋め込み次元に射影する MLP。出力は1トークンとしてトークン列に追加される。

### ActionHead インターフェース

```python
class ActionHead(ABC):
    """すべてのアクションヘッドの共通インターフェース"""

    @abstractmethod
    def compute_loss(self, features: Tensor, actions: Tensor, **kwargs) -> dict:
        """
        学習時: 損失とメトリクスを返す。
        features: [B, D] LLM バックボーンからの特徴ベクトル
        actions:  [B, T, action_dim] 正解アクション
        returns:  {"loss": Tensor, ...追加メトリクス}
        """

    @abstractmethod
    def predict(self, features: Tensor, **kwargs) -> Tensor:
        """
        推論時: 予測アクションを返す。
        returns: [B, T, action_dim]
        """
```

**`**kwargs` の用途:**
- ACT: 学習時のエンコーダ追加入力
- Flow Matching: 推論時のデノイジングステップ数
- UniAct: コードブック関連パラメータ

### アクションヘッド一覧

| ヘッド | 方式 | 損失関数 | 初期実装 |
|--------|------|----------|----------|
| MLPHead | 決定的回帰 | MSE | Yes |
| ACTHead | CVAE + チャンキング | 再構成誤差 + KL | No（後日） |
| FlowMatchingHead | 条件付きフローマッチング | CFM 損失 | No（後日） |
| UniActHead | VQ コードブック + ロボット固有デコーダ | VQ 損失 + 復元損失 | No（後日） |

## データパイプライン

### データ形式: LeRobot

LeRobot 形式（HuggingFace Datasets ベース）を採用。Open X-Embodiment の公開データセットも LeRobot 形式で HuggingFace Hub に変換済みのものを利用可能（例: `lerobot/bridge_v2`）。自前データも同形式で作成すれば統一的に扱える。

### VLADataset

```python
class VLADataset:
    """LeRobot データセットのラッパー"""

    def __getitem__(self, idx) -> dict:
        return {
            "images": [Tensor, ...],       # カメラ数分のリスト, 各 [C, H, W]
            "instruction": str,             # タスク指示テキスト
            "proprio": Tensor,              # [7] EEF 位置+回転+グリッパー
            "actions": Tensor,              # [T, 7] 正解アクション
        }
```

**設定で切り替え可能な項目:**
- カメラ名のリスト（例: `["observation.images.top", "observation.images.wrist"]`）
- チャンク長 T（1ステップ or チャンキング）
- 画像前処理パラメータ

### データ前処理

- 画像: リサイズ、正規化（`transforms.py` に分離）
- アクション: mean/std 正規化（データセット単位で管理）

## 学習パイプライン

### 学習ループ

```python
for batch in dataloader:
    features = policy.encode(batch)
    loss_dict = policy.action_head.compute_loss(features, batch["actions"])
    loss_dict["loss"].backward()
    optimizer.step()
```

### 学習戦略（設定で切り替え）

| 戦略 | 対象 | 用途 |
|------|------|------|
| フル FT | 全パラメータ | 精度重視 |
| LoRA | LLM に LoRA 適用、ヘッド+ProprioEncoder はフル学習 | VRAM 節約 |

LoRA の適用には HuggingFace PEFT ライブラリを使用。

### 学習設定

- Optimizer: AdamW
- スケジューラ: cosine with warmup
- Mixed precision: bf16
- マルチ GPU: HuggingFace Accelerate
- チェックポイント保存・再開対応

## 推論

### 量子化オプション（設定で切り替え）

| 精度 | ライブラリ | 用途 |
|------|-----------|------|
| bf16 | — | デフォルト |
| 8bit | bitsandbytes | サーバー推論 |
| 4bit | bitsandbytes | エッジデバイス |

HuggingFace Transformers の `BitsAndBytesConfig` で設定。アーキテクチャ変更は不要。

## 設定管理

YAML ベースの設定ファイルで以下を管理:

- `model_name`: バックボーンモデル（サイズ変更はここだけ）
- `cameras`: カメラ名リスト（数の変更はここだけ）
- `action_head`: ヘッド種類とパラメータ
- `chunk_size`: チャンク長（1 = 1ステップ予測）
- `action_dim`: アクション次元（デフォルト 7）
- `training.strategy`: `"full"` or `"lora"`
- `training.lora`: LoRA パラメータ（r, alpha, target_modules）
- `inference.quantization`: `null` / `"8bit"` / `"4bit"`

## 初期実装スコープ

- [x] VLAPolicy（Gemma 4 E2B 統合）
- [x] ProprioEncoder
- [x] ActionHead ABC
- [x] MLPHead（1ステップ + チャンキング対応）
- [x] VLADataset（LeRobot ラッパー）
- [x] 学習パイプライン（フル FT + LoRA）
- [x] YAML 設定管理
- [x] 学習・評価スクリプト

## 将来拡張

- ACTHead, FlowMatchingHead, UniActHead の実装
- 推論時量子化（8bit / 4bit）
- マルチロボット対応（UniAct コードブック共有）
