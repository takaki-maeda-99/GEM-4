"""
Gemma 4 E2B backbone 用の定数 (Stage 1)。

通常の `constants.py` (Qwen2.5-0.5B 用) と併存し、backbone が Gemma 4 のときだけこちらを使う。
ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM, NUM_TOKENS, NormalizationType 等の
backbone 非依存な値は `constants.py` からそのまま import して使う。
"""

# ============================================================
# Gemma 4 E2B 固有の値
# ============================================================

# Action token special IDs:
# Gemma 4 の vocab に 6227 個の <unusedX> 予約トークンがあり、
# その中から video_token_id (258884) の直後の 64 個を action query 用に割り当てる。
#
#   258885-258948 (`<unused2968>` 〜 `<unused3031>`)
#
# これらの ID は:
#   - 標準 tokenizer のどの added_token とも重複しない
#   - 全て vocab 内の単一 token として embedding lookup 可能
#   - 意味的 token (語彙) とも重複しない
#
# Qwen2.5-0.5B 用の旧値: 151386
ACTION_TOKEN_BEGIN_IDX = 258885
NUM_ACTION_TOKENS      = 64                # Action query 個数 (= VLA-Adapter §2.3 の慣例)

# Vision placeholder ID range: 258949 〜 259460 (連続 512 個、全て <unusedX>).
# Action range (258885-258948) の直後に連続して確保。
# tokenizer での検証済み: 258949=<unused3032>, 259460=<unused3543>、全て <unusedXXXX> パターン。
VISION_PLACEHOLDER_BEGIN_IDX = 258949
# DEPRECATED (Task 6 rev 3): max_soft_tokens が YAML で可変になったため、この constant は
# Task 14 の data loader migration で完全に model.num_vision_tokens (VLAAdapterGemma4 attr)
# に置換される予定。それまでの暫定値として max_soft_tokens=280 の num_soft_tokens_per_image
# 値 (実測 256) を置く。Task 6 スコープ外の参照箇所 (multi_dataset_loader.py,
# eval_libero_gemma4.py, test_08_data_pipeline.py 等) は Task 14 で migration する。
NUM_VISION_TOKENS            = 256
NUM_VISION_PLACEHOLDERS      = 256  # backward-compat alias

# Proprio placeholder ID (1 個)。vision placeholder range の直後。
PROPRIO_PLACEHOLDER_IDX = 259461

# Native Gemma 4 image token ID (multimodal pretrain で使われる、`<|image|>` token).
# 2026-04-26: vision placeholder mode ablation 用。
# `VLA_VISION_PLACEHOLDER_MODE=image_token` 時に loader が unique <unused> ID 列の代わりに
# IMAGE_TOKEN_ID を num_vision_tokens 個並べる。PLE が pretrain 由来の値になる仮説検証。
# tokenizer 検証: ID=258880 == `<|image|>` (added_tokens_decoder 内)
IMAGE_TOKEN_ID = 258880

# End-of-sequence token ID (Gemma 4 の eos).
# Qwen の `</s>` = 2 とは別物。Gemma の config を確認:
#   cfg.text_config.eos_token_id == 1
#   cfg.text_config.bos_token_id == 2
# VLA-Adapter の STOP_INDEX は generation 時の終端判定で使われる。Stage 1 (学習のみ) では
# この値は使わないが、Stage 2 推論で必要になるため定義しておく。
STOP_INDEX = 1

# ============================================================
# Gemma 4 E2B アーキテクチャのメタデータ (便宜上ここに集約)
# ============================================================

# Backbone に関する config-level fact. コードからは参照しないが
# 移植時のレビュー用に値を明示しておく。
GEMMA4_E2B_META = {
    "hidden_size": 1536,          # text_config.hidden_size
    "num_hidden_layers": 35,
    "num_attention_heads": 8,
    "num_key_value_heads": 1,     # 極端な MQA
    "head_dim_sliding": 256,
    "head_dim_global": 512,       # dual head_dim (G2)
    "sliding_window": 512,
    "num_kv_shared_layers": 20,   # G1: layer 15-34 が KV 共有参照
    "hidden_size_per_layer_input": 256,  # G3: PLE
    "vocab_size": 262144,
    "max_position_embeddings": 131072,
    "attention_pattern": "4 sliding + 1 full, repeated 7x",
    "full_layer_indices": [4, 9, 14, 19, 24, 29, 34],
    "tie_word_embeddings": True,  # lm_head と input embedding が共有
}
