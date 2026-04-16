# ACTHead 設計仕様書

## 概要

ACT (Action Chunking with Transformers) 方式のアクションヘッドを実装する。Transformer デコーダによるチャンキング予測（複数ステップ同時予測）と Temporal Ensemble による推論時のアクション平滑化を行う。初期実装では CVAE を省略し、チャンキング + Transformer デコーダのみで MLPHead との性能差を検証する。

## アーキテクチャ

### 学習時

```
features [B, N_act, D]
  → Linear projection → [B, N_act, d_model]           (ソースとして使用)

chunk_queries [chunk_size, d_model]                     (学習可能パラメータ)
  → expand to [B, chunk_size, d_model]

Transformer デコーダ:
  query = chunk_queries  [B, chunk_size, d_model]
  memory = projected features  [B, N_act, d_model]
  → self-attention (query 間)
  → cross-attention (query → features)
  → × num_layers 回

→ Linear → [B, chunk_size, action_dim]
→ MSE loss vs GT action chunk [B, chunk_size, action_dim]
```

### 推論時

```
features [B, 1, D]
  → 同じデコーダ → [B, chunk_size, action_dim]  (20ステップ分の予測)
  → Temporal Ensemble: 過去の予測と指数減衰重み付き平均
  → [B, 1, action_dim]  (1ステップ分のアクションを出力)
```

### Transformer デコーダ パラメータ

| パラメータ | デフォルト値 | 説明 |
|-----------|------------|------|
| `d_model` | 256 | デコーダの隠れ次元 |
| `nhead` | 4 | Attention ヘッド数 |
| `num_layers` | 2 | デコーダレイヤー数 |
| `chunk_size` | 20 | チャンク長（予測ステップ数） |
| `dim_feedforward` | 1024 | FFN の中間次元 |

### Temporal Ensemble

推論時に過去 `chunk_size` 回分の予測を保持し、指数減衰重みで加重平均する。

```
時刻 t の最終アクション = Σ_k w_k * prediction_{t-k}[k]
                         k=0..min(t, chunk_size-1)

w_k = exp(-m * k)  (m = temporal_ensemble_m, デフォルト 0.01)
```

- `reset_ensemble()` でエピソード開始時にバッファをクリア
- バッファは `predict()` 内部で管理

## ActionHead インターフェース

```python
class ACTHead(ActionHead):
    def __init__(
        self,
        input_dim: int,           # LLM hidden dim (1536)
        action_dim: int = 7,
        chunk_size: int = 20,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        temporal_ensemble_m: float = 0.01,
    ): ...

    def compute_loss(self, features: Tensor, actions: Tensor, **kwargs) -> dict:
        """
        features: [B, N_act, D]
        actions: [B, chunk_size, action_dim]  ← chunk 分の GT アクション
        returns: {"loss": mse_loss, "mse_loss": mse_loss}
        """

    def predict(self, features: Tensor, **kwargs) -> Tensor:
        """
        features: [B, N_act, D]
        returns: [B, 1, action_dim]  ← temporal ensemble 適用後の1ステップ
        """

    def reset_ensemble(self):
        """エピソード開始時にバッファをクリア"""
```

## 設定ファイル

`configs/libero_spatial_act.yaml` の差分（MLPとの違い）:

```yaml
chunk_size: 20  # 1 → 20

action_head:
  type: "act"
  d_model: 256
  nhead: 4
  num_layers: 2
  dim_feedforward: 1024
  temporal_ensemble_m: 0.01
  gripper_as_binary: false
```

## 変更ファイル

### 新規
- `vla_gemma4/model/action_heads/act_head.py` — ACTHead 実装
- `tests/test_act_head.py` — テスト
- `vla_gemma4/configs/libero_spatial_act.yaml` — ACT 用設定

### 変更
- `vla_gemma4/model/vla_policy.py` — `_build_action_head` に `"act"` タイプ追加
- `scripts/eval_libero.py` — エピソード開始時に `reset_ensemble()` 呼び出し

### 変更なし
- `ActionHead` ABC
- `VLADataset` — `chunk_size` は設定で切り替え
- `train.py`, `VLATrainer`

## 将来拡張

- CVAE エンコーダの追加（KL 正則化でマルチモーダル行動分布の学習）
- `num_action_tokens` を `chunk_size` に合わせて増やすパターンの検証
