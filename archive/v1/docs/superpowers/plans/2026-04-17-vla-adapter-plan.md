# VLA-Adapter 風アーキテクチャ実装計画

**日付**: 2026-04-17
**目的**: 現在の `VLAModel`（Bridge + Flow Matching）を、[VLA-Adapter](https://vla-adapter.github.io/) 論文の構造に近づけて段階的に書き換える。

---

## 現状の問題

今の実装で分かっていること：
- Frozen VLM + Bridge + Flow Matching Head で構築
- 1 サンプル overfit は Bridge 修正後に通る（MSE 0.0005）
- 10 エピソードの overfit は **上手く行かない**（flow matching loss が振動、予測 MSE 改善遅い）
- VLM features のサンプル間 cosine 0.99、Bridge 通すと 0.9999 に均一化

---

## VLA-Adapter との主な違い

論文の定式（per Policy layer τ）:

```
Â^τ = concat([
    CA₁(Ã^τ, σ₁(C^R_τ)) · tanh(g),            # Raw features (per-layer)
    CA₂(Ã^τ, σ₂([C^AQ_τ, σ₀(P)])),            # ActionQuery features + proprio
    SA(Ã^τ, Ã^τ),                             # 自己注意
])
```

| 項目 | VLA-Adapter | 現状 |
|---|---|---|
| 損失 | **L1** | Flow matching |
| ActionQuery の位置 | **VLM 内部にトークンとして注入** | VLM 外部の learnable 埋め込み |
| Cross-attention | **2つ**（Raw / AQ） | 1つ（全層 concat） |
| Policy の層数 | VLM と同じ数、層 τ ↔ VLM 層 τ | 4 層、全層 concat |
| 学習可能ゲート | `tanh(g)`（初期 0） | なし |
| Proprio | AQ features と concat → CA₂ に入る | 独立した token として最後に concat |

---

## 段階的実装計画

### Stage A: Flow Matching → L1（最小変更で効果検証）

**やること**:
- `FlowMatchingHead` を `L1ChunkHead`（MLP or 小さな Transformer decoder）に差し替え
- Bridge と Proprio の構造は現状維持
- 他のハイパラは現状維持

**成功基準**:
- 10 エピソードの overfit が MSE < 0.05 に落ちる（現状 0.2 以上）
- 学習 loss が単調減少（flow matching の振動なし）

**失敗したら**: Stage B に進むべきかもう一度切り分け。

---

### Stage B: Bridge を VLA-Adapter 2-branch 構造に

**やること**:
- Bridge に 2 つの Cross-Attention を入れる:
  - `CA_raw`: Q = action latent, KV = raw VLM hidden states（全トークン）
  - `CA_aq`: Q = action latent, KV = ActionQuery tokens（後述）+ proprio
- Self-attention はそのまま
- 出力を concat（チャンネル方向）→ FFN → 次の layer へ
- 学習可能ゲート `g`（初期 0, `tanh(g)` で raw を gating）

**ActionQuery の扱い（Stage B 版）**:
- Stage B では簡略化: ActionQuery は VLM 外部の learnable embedding のまま
- `C^AQ` = 現在の ActionQuery そのもの
- Policy 各層でこの AQ tokens に cross-attend

**成功基準**:
- Stage A と同等以上の overfit
- 初期化時点の Bridge 出力 cosine が < 0.95（サンプル識別性あり）

---

### Stage C: Per-layer correspondence

**やること**:
- VLM の層を Policy 層と同数選ぶ（例: 6 層 VLM → 6 層 Policy）
- Policy 層 τ の CA₁ は VLM 層 τ の hidden states を見る
- Policy 層 τ の CA₂ は VLM 層 τ の AQ tokens を見る（現状は全層同じ AQ）

**成功基準**:
- Stage B より loss 下がる or 同等
- LIBERO rollout で成功例が出る

---

### Stage D: ActionQuery を VLM 内部に注入（本命）

**やること**:
- ActionQuery を Gemma4 の input sequence に soft token として追加（学習可能 embedding）
- VLM forward で AQ tokens も他のトークンと相互作用する
- Policy は各層の AQ positions の hidden states を抽出

**難度**:
- `inputs_embeds` 経由で VLM forward、PLE 計算の扱い等、以前ハマったところを再訪
- frozen VLM なので gradient は AQ の embedding テーブルにだけ流れる

**成功基準**:
- LIBERO 単タスクで rollout 成功率 > 50%
- full LIBERO-Spatial で success rate が前実装を超える

---

## 中断条件

各 Stage で **overfit（学習データ丸暗記）すら通らない** 場合、その Stage のアーキテクチャに問題がある。次 Stage に進まずに修正。

---

## 参考

- 論文: https://arxiv.org/abs/2509.09372
- プロジェクトページ: https://vla-adapter.github.io/
- GitHub: https://github.com/OpenHelix-Team/VLA-Adapter
- 過去トラブル記録: `docs/troubleshooting.md`
