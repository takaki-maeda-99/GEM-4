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
- **HuggingFace**: `AutoModelForImageTextToText.from_pretrained("google/gemma-4-E2B-it")`
- バックボーンサイズは `model_name` の変更のみで切り替え可能（例: `google/gemma-4-4B-it`）

### アクション空間

- **形式**: EEF デルタポーズ（相対座標）
- **次元**: 7 (Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper)
  - 位置・回転（dim 0-5）: 連続値、MSE 損失
  - グリッパー（dim 6）: [0, 1] の連続値（0=閉、1=開）、BCE 損失で個別に扱う
- **出力形状**: `[B, T, 7]` — T=1 で1ステップ予測、T>1 でチャンキング予測

### 特徴抽出

学習可能な `[ACT]` トークン（特殊トークン埋め込み）をトークン列の末尾に追加し、LLM 最終層のその位置の隠れ状態をアクションヘッドへの入力とする。

- **トークン数**: `num_action_tokens` で設定可能（デフォルト 1）
  - MLPHead: 1 トークンで十分（出力を reshape して `[B, T, 7]` を生成）
  - ACTHead: チャンクサイズ分のクエリトークンが必要になる可能性があるため、ヘッドに応じて変更可能にする
- **特徴形状**: `[B, N_act, D]`（N_act = num_action_tokens）
  - N_act=1 のとき実質 `[B, 1, D]`、ヘッド側で squeeze 可能

### 時間的コンテキスト

初期実装では**単一タイムステップの観測**のみを入力とする（観測履歴なし）。これは設計のシンプルさを優先した意図的な選択。将来的に過去フレームの履歴を入力に追加する拡張は、トークン列に過去の画像トークンを追加する形で対応可能。

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
│   ├── normalizer.py           # Normalizer: アクション正規化・逆正規化
│   └── transforms.py           # 画像前処理・データ拡張
├── training/
│   └── trainer.py              # 学習ループ
├── scripts/
│   ├── train.py                # 学習スクリプト
│   └── eval.py                 # 評価スクリプト（オフライン評価 + 推論ループ）
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
class ActionHead(nn.Module, ABC):
    """すべてのアクションヘッドの共通インターフェース（nn.Module を継承し、パラメータ管理・デバイス移動に対応）"""

    @abstractmethod
    def compute_loss(self, features: Tensor, actions: Tensor, **kwargs) -> dict:
        """
        学習時: 損失とメトリクスを返す。
        features: [B, N_act, D] LLM バックボーンからの特徴ベクトル
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
| MLPHead | 決定的回帰 | MSE（位置・回転）+ BCE（グリッパー） | Yes |
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

### 言語指示の取得

- LeRobot データセットの `language_instruction` フィールドから取得
- フィールドが存在しない場合は設定ファイルでデフォルトの指示テキストを指定可能（例: `"manipulation task"`）
- 将来的に指示文のパラフレーズ拡張も検討可能

### データ前処理

- 画像: Gemma 4 の `AutoProcessor` による前処理を使用（リサイズ・正規化はプロセッサが担当）。ビジュアルトークンバジェットは設定で指定。
- アクション正規化: `Normalizer` クラスで管理
  - 学習データから mean/std を事前計算し、JSON ファイルとして保存
  - `VLADataset` 内で正規化を適用（`__getitem__` 時）
  - 推論時は同じ `Normalizer` で逆正規化してロボットへの生のデルタ指令に変換
  - 正規化統計は学習データセットごとに管理（マルチデータセット学習時はデータセット単位で適用）

## 学習パイプライン

### 学習ループ

```python
for batch in dataloader:
    # encode() は微分可能 — 勾配はアクションヘッド → LLM バックボーン → ViT まで流れる（E2E）
    features = policy.encode(batch)
    loss_dict = policy.action_head.compute_loss(features, batch["actions"])
    loss_dict["loss"].backward()
    optimizer.step()
```

### 学習戦略（設定で切り替え）

| 戦略 | 対象 | 用途 |
|------|------|------|
| フル FT | 全パラメータ（ViT + LLM + ヘッド + ProprioEncoder） | 精度重視 |
| LoRA | LLM に LoRA 適用、ViT はフリーズ、ヘッド + ProprioEncoder はフル学習 | VRAM 節約 |

LoRA の適用には HuggingFace PEFT ライブラリを使用。LoRA モード時は ViT をフリーズすることで VRAM を大幅に削減する。

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

## 評価

### オフライン評価

- テストセットに対するアクション予測精度（MSE、L1 誤差）
- グリッパー予測精度（accuracy、F1）
- `eval.py` で実行、結果を JSON/CSV で出力

### オンライン評価（将来対応）

- シミュレーション環境（SIMPLER、ManiSkill 等）でのタスク成功率
- 実機でのタスク成功率
- 初期実装スコープ外だが、推論ループ（観測取得 → モデル推論 → アクション送信）は `eval.py` に実装しておく

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
