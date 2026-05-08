"""
action_heads.py

Implementations of various action heads, which serve as alternatives to VLM sequential token prediction.
"""

import math
import torch
import torch.nn as nn
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX, NUM_TOKENS



def learnable_random_perturbations(seq_len, dim, device, dtype):
    random_perturbations = nn.Parameter(torch.zeros(seq_len, dim, device=device, dtype=dtype))
    nn.init.normal_(random_perturbations, mean=0.0, std=0.02)
    return random_perturbations



class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_task_tokens=512,
        use_pro_version=False,
        use_xvla_style=False,       # 2026-04-24 #014: X-VLA 準拠 self-attn pool 単一化 (Bridge cross-attn 撤去)
        use_proper_ffn=False,       # 2026-04-24 #016: pre-LN + 4× FFN + dual residual
        num_blocks=24,              # 2026-04-25 #021: configurable。Gemma4 E2B=35 層対応など用。
    ):
        super().__init__()
        self.num_task_tokens = num_task_tokens
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.use_xvla_style = use_xvla_style
        self.num_blocks = num_blocks
        self.model = MLPResNet(
            num_blocks=num_blocks,
            input_dim=input_dim*ACTION_DIM,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
            use_pro_version=use_pro_version,
            use_xvla_style=use_xvla_style,
            use_proper_ffn=use_proper_ffn,
            )

    def predict_action(
            self,
            actions_hidden_states,
            proprio=None,
            proprio_projector=None,
            phase="Inference",
            h_w=None,
            h_sp=None,
            h_v=None,        # 2026-04-24 #014: X-VLA 流、scene final-layer tokens (B, N_v, llm_dim)
            h_w_bridge=None, # 2026-04-24 #015 option B: wrist per-layer (B, num_layers, K_w, llm_dim)
            ):
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device

        proprio = proprio.reshape(batch_size, -1).to(torch.bfloat16)  # (bsz, proprio_dim)
        proprio_features = proprio_projector(proprio)  # (bsz, llm_dim)
        proprio_features = proprio_features.unsqueeze(dim=1)  # (bsz, 1, llm_dim)

        task_hidden_states = actions_hidden_states[:, :, :self.num_task_tokens, :]
        actions_hidden_states = actions_hidden_states[:, :, self.num_task_tokens:, :]

        cond_actions_hidden_states = torch.zeros(
            (batch_size, self.action_dim * NUM_ACTIONS_CHUNK, self.hidden_dim),
            device=device, dtype=actions_hidden_states.dtype
        ).detach()

        rearranged_actions_hidden_states = cond_actions_hidden_states.reshape(
            batch_size, NUM_ACTIONS_CHUNK, -1
        )  # (batch, chunk_len, action_dim * hidden_dim)

        if phase == "Training":
            batch_size, seq_len, dim = rearranged_actions_hidden_states.shape
            random_perturbations = learnable_random_perturbations(seq_len, dim, device=rearranged_actions_hidden_states.device, dtype=rearranged_actions_hidden_states.dtype)
            rearranged_actions_hidden_states = (rearranged_actions_hidden_states + random_perturbations) # (1, seq_len, dim)

        action = self.model(
            rearranged_actions_hidden_states,
            h_a=actions_hidden_states,
            p=proprio_features,
            h_t=task_hidden_states,
            h_w=h_w,
            h_sp=h_sp,
            h_v=h_v,
            h_w_bridge=h_w_bridge,
            )

        return action
    

class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(
            self,
            num_blocks,
            input_dim,
            hidden_dim,
            output_dim,
            use_pro_version=False,
            use_xvla_style=False,
            use_proper_ffn=False,
            ):

        super().__init__()
        # 2026-04-24 #014 revised: use_xvla_style=True は **option (a) = Bridge 維持 + scene concat**
        # 2026-04-24 #016: use_proper_ffn=True で各 block を標準 pre-LN Transformer block に昇格
        self.use_xvla_style = use_xvla_style
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()

        for _ in range(num_blocks):
            if use_pro_version:
                self.mlp_resnet_blocks.append(
                    MLPResNetBlock_Pro(dim=hidden_dim, use_proper_ffn=use_proper_ffn)
                )
            else:
                self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))

        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)


    def forward(self, x, h_a=None, h_t=None, p=None, h_w=None, h_sp=None, h_v=None, h_w_bridge=None):
        """Two modes (2026-04-24 #014 revised = option (a)):
          use_xvla_style=False (Bridge default): action + h_w + h_sp concat to self-attn pool,
            blocks cross-attend to h_a/h_t/p (VLA-Adapter Bridge 形)。
          use_xvla_style=True (Bridge + scene concat): 上記に加えて h_v (scene final-layer) も
            self-attn pool に concat。Bridge cross-attn は **そのまま維持**。Bridge baseline
            (61%) を壊さずに wrist attention starvation を緩和 (scene と wrist が並列に action
            queries から self-attn される)。

        Args:
            x:   (B, NUM_ACTIONS_CHUNK, hidden_dim) action latent (after fc1)
            h_a: (B, num_layers, K_a, hidden_dim) Bridge adapter hidden (両モードで使用)
            h_t: (B, num_layers, K_t, hidden_dim) Bridge task hidden (両モードで使用)
            p:   (B, 1, hidden_dim) proprio (cross-attn 側、pool には入れない)
            h_w: (B, 49, hidden_dim) wrist tokens (optional、両モードで pool に concat)
            h_sp:(B, 32, hidden_dim) soft prompt tokens (optional、両モードで pool に concat)
            h_v: (B, N_v, hidden_dim) scene final-layer tokens (use_xvla_style=True のみ使用)
        Returns:
            (B, NUM_ACTIONS_CHUNK, output_dim)
        """
        action_len = x.shape[1]  # remember action token count for trim

        x = self.layer_norm1(x)
        x = self.fc1(x)
        x = self.relu(x)

        # 2026-04-24 #014 revised: option (a) = Bridge 維持 + scene concat
        # use_xvla_style=True: x に h_v (scene final-layer) を追加 concat → self-attn pool 拡張
        # use_xvla_style=False: 既存 Bridge (scene は h_t cross-attn 経由のみ)
        # 両モードで h_a/h_t cross-attn は block 内で実行 (Bridge stream 維持)。
        if self.use_xvla_style and h_v is not None:
            x = torch.cat([x, h_v], dim=1)
        # 2026-04-24 #017: use_wrist_bridge=True の時は wrist は専用 cross-attn stream に入るので
        # self-attn pool に h_w を concat しない (wrist が 2 経路に重複していた redundancy 解消)。
        use_wrist_bridge_active = (h_w_bridge is not None)
        if h_w is not None and not use_wrist_bridge_active:
            x = torch.cat([x, h_w], dim=1)
        if h_sp is not None:
            x = torch.cat([x, h_sp], dim=1)

        for i, block in enumerate(self.mlp_resnet_blocks):
            # 2026-04-24 #015 option B: h_w_bridge が渡ってれば per-layer で cross-attn
            h_w_l = h_w_bridge[:, i + 1, :] if h_w_bridge is not None else None
            x = block(x, h_t=h_t[:, i + 1, :], h_a=h_a[:, i + 1, :], p=p, h_w_l=h_w_l)

        # Trim: keep only action positions
        x = x[:, :action_len, :]

        x = self.layer_norm2(x)
        x = self.fc2(x)
        return x



def apply_rope(q, k, cos, sin):
    """
    RoPE:
    q, k: (B, H, T, D)   # D must be an even number
    cos/sin: (T, D)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
    sin = sin.unsqueeze(0).unsqueeze(0)


    def rotate_half(x):
        # Swap even and odd dimensions and flip the signs
        x1 = x[..., ::2]   # Even subdimension
        x2 = x[..., 1::2]  # odd subdimension

        return torch.stack((-x2, x1), dim=-1).reshape_as(x)


    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)

    return q_rot, k_rot



class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        """
        dim = head_dim
        """
        super().__init__()
        assert dim % 2 == 0, "RoPE head_dim must be an even number"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (T, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)            # (T, dim)
        return emb.cos().to(dtype), emb.sin().to(dtype)



class MLPResNetBlock(nn.Module):
    """
    One residual MLP block with cross-attention conditioning.

    This block applies multi-head attention over:
      - token features (self-attention),
      - task-related hidden states (h_t),
      - action/proprioception-related hidden states (h_a, p).
    The outputs are combined via a gating mechanism, projected back to the
    hidden dimension, and passed through a small feedforward sub-network with
    residual connection.

    Args:
        dim (int): Dimensionality of the hidden features. Must be divisible by num_heads.

    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
        h_t (torch.Tensor, optional): Task-related hidden states of shape
                                      (batch_size, K, hidden_dim).
        h_a (torch.Tensor, optional): Action-related hidden states of shape
                                      (batch_size, 1, hidden_dim).
        p (torch.Tensor, optional): Additional conditioning features
                                    (e.g., proprioception), shape (batch_size, 1, hidden_dim).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim).
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # Main feedforward network
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

        self.num_heads = 8
        self.head_dim = dim // self.num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        self.gating_factor = nn.Parameter(torch.zeros(1))



    def forward(self, x, h_t=None, h_a=None, p=None):
        """
        x: (batch_size, seq_len, hidden_dim)
        h, t, p: (batch_size, 1, hidden_dim) or None
        """

        g = self.gating_factor
        ratio_g = nn.Tanh()(g)

        conditions = []
        if h_a is not None:
            conditions.append(h_a)
        if p is not None:
            conditions.append(p)

        h = torch.cat(conditions, dim=1)  # (batch_size, cond_len, hidden_dim)

        B = x.size(0)
        T = x.size(1)
        C = x.size(2)
        K_t = h.size(1)
        K = h_t.size(1)

        task_k = h
        task_v = h

        adapter_k = h_t
        adapter_v = h_t

        q_1 = self.q_proj(x) # (B, T, C)
        k_tokens = self.k_proj(x)             # (B, T, C)
        v_tokens = self.v_proj(x)             # (B, T, C)
        k_task = self.k_proj(task_k)    # (B, K, C)
        v_task = self.v_proj(task_v)    # (B, K, C)

        k_adapter = self.k_proj(adapter_k)    # (B, K, C)
        v_adapter = self.v_proj(adapter_v)    # (B, K, C)

        # (B, seq_len, C) -> (B, num_heads, seq_len, head_dim)
        q_1 = q_1.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        k_tokens = k_tokens.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v_tokens = v_tokens.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_task = k_task.view(B, K_t, self.num_heads, self.head_dim).transpose(1, 2)
        v_task = v_task.view(B, K_t, self.num_heads, self.head_dim).transpose(1, 2)

        k_adapter = k_adapter.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v_adapter = v_adapter.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores_tokens = torch.matmul(q_1, k_tokens.transpose(-2, -1)) # (B, H, T, T)
        attn_scores_task = torch.matmul(q_1, k_task.transpose(-2, -1)) * 1 # (B, H, T, K)
        attn_scores_adapter = torch.matmul(q_1, k_adapter.transpose(-2, -1)) * ratio_g # (B, H, T, K)

        attn_scores = torch.cat([attn_scores_tokens, attn_scores_task, attn_scores_adapter], dim=-1) # (B, H, T, T+K)
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1) # (B, H, T, T+K)

        v_combined = torch.cat([v_tokens, v_task, v_adapter], dim=2) # (B, H, T+K, head_dim)
        output = torch.matmul(attn_weights, v_combined) # (B, H, T, head_dim)

        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        x = self.ffn(output + x) 

        return x



class MLPResNetBlock_Pro(nn.Module):
    """One MLP ResNet block with separate projections for self, adapter, task + RoPE.

    2026-04-24 #016: use_proper_ffn=True で pre-LN + 4× FFN + dual residual (標準 Transformer block) に昇格。
    use_proper_ffn=False (default): 従来の attn + ffn (Linear D→D + ReLU、residual 無し) を維持 (backward compat)。
    """

    def __init__(self, dim, num_heads=8, use_proper_ffn=False, mlp_ratio=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_proper_ffn = use_proper_ffn

        if use_proper_ffn:
            # 標準 pre-LN Transformer block
            self.norm1 = nn.LayerNorm(dim)          # attn 前 LN
            self.norm2 = nn.LayerNorm(dim)          # FFN 前 LN
            self.ffn_up = nn.Linear(dim, dim * mlp_ratio)     # 4× expansion
            self.ffn_down = nn.Linear(dim * mlp_ratio, dim)   # output projection
            self.ffn = None   # legacy FFN は使わない
        else:
            # Legacy (pre #016): 貧弱 FFN, residual 置換
            self.ffn = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
                nn.ReLU(),
                )

        # Q (from x only)
        self.q_proj = nn.Linear(dim, dim)

        # Self-Attention: K, V
        self.k_self = nn.Linear(dim, dim)
        self.v_self = nn.Linear(dim, dim)

        # Adapter cross-attention: K, V
        self.k_adapter = nn.Linear(dim, dim)
        self.v_adapter = nn.Linear(dim, dim)

        # Task cross-attention: K, V
        self.k_task = nn.Linear(dim, dim)
        self.v_task = nn.Linear(dim, dim)

        # Wrist cross-attention: K, V (2026-04-24 #015 option B、h_w_l 渡された時のみ使う)
        self.k_wrist = nn.Linear(dim, dim)
        self.v_wrist = nn.Linear(dim, dim)

        self.o_proj = nn.Linear(dim, dim)

        # gating (task stream 専用、legacy)
        self.gating_factor = nn.Parameter(torch.zeros(1))
        # wrist gating: tanh(0)=0 で init、学習で段階的に wrist 寄与を引き上げる (warmup 相当)
        self.gating_factor_wrist = nn.Parameter(torch.zeros(1))

        # RoPE
        self.rope = RotaryPositionEmbedding(self.head_dim)


    def forward(self, x, h_a=None, h_t=None, p=None, h_w_l=None):
        """
        h_a:   adapter tokens (Bridge)
        h_t:   task tokens (Bridge scene)
        p:     proprio conditioning vector
        h_w_l: (B, K_w, dim) wrist layer tokens (option B)、None なら wrist stream skip
        """
        g = self.gating_factor
        ratio_g = torch.tanh(g)

        # concat h_a and p
        h_adapter = torch.cat((h_a, p),dim=1)

        h_task = h_t
        B, T, C = x.shape
        K_a = h_adapter.size(1) if h_a is not None else 0
        K_t = h_task.size(1) if h_task is not None else 0
        use_wrist = h_w_l is not None
        K_w = h_w_l.size(1) if use_wrist else 0

        # Proper FFN 経路では attn 入力に pre-LN (norm1) 適用
        x_attn_in = self.norm1(x) if self.use_proper_ffn else x

        # Q
        q_1 = self.q_proj(x_attn_in)

        # self tokens (from x_attn_in; Pro では norm1(x) 経由)
        k_tokens = self.k_self(x_attn_in)
        v_tokens = self.v_self(x_attn_in)

        # adapter tokens
        k_adapter = self.k_adapter(h_adapter)
        v_adapter = self.v_adapter(h_adapter)

        # task tokens
        k_task = self.k_task(h_task)
        v_task = self.v_task(h_task)

        # wrist tokens (option B)
        if use_wrist:
            k_wrist = self.k_wrist(h_w_l)
            v_wrist = self.v_wrist(h_w_l)


        # reshape -> multi-head
        def reshape_heads(t, B, L):
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)


        q_1 = reshape_heads(q_1, B, T)
        k_tokens, v_tokens = reshape_heads(k_tokens, B, T), reshape_heads(v_tokens, B, T)
        k_adapter, v_adapter = reshape_heads(k_adapter, B, K_a), reshape_heads(v_adapter, B, K_a)
        k_task, v_task = reshape_heads(k_task, B, K_t), reshape_heads(v_task, B, K_t)
        if use_wrist:
            k_wrist, v_wrist = reshape_heads(k_wrist, B, K_w), reshape_heads(v_wrist, B, K_w)

        # RoPE
        cos_main, sin_main = self.rope(seq_len=T, device=x.device, dtype=x.dtype)
        q_1, k_tokens = apply_rope(q_1, k_tokens, cos_main, sin_main)
        cos_a, sin_a = self.rope(seq_len=K_a, device=x.device, dtype=x.dtype)
        _, k_adapter = apply_rope(k_adapter, k_adapter, cos_a, sin_a)
        cos_t, sin_t = self.rope(seq_len=K_t, device=x.device, dtype=x.dtype)
        _, k_task = apply_rope(k_task, k_task, cos_t, sin_t)
        if use_wrist:
            cos_w, sin_w = self.rope(seq_len=K_w, device=x.device, dtype=x.dtype)
            _, k_wrist = apply_rope(k_wrist, k_wrist, cos_w, sin_w)

        # attention scores
        attn_scores = [torch.matmul(q_1, k_tokens.transpose(-2, -1))]
        attn_scores.append(torch.matmul(q_1, k_adapter.transpose(-2, -1)))
        attn_scores.append(torch.matmul(q_1, k_task.transpose(-2, -1)) * ratio_g)
        if use_wrist:
            ratio_g_wrist = torch.tanh(self.gating_factor_wrist)
            attn_scores.append(torch.matmul(q_1, k_wrist.transpose(-2, -1)) * ratio_g_wrist)
        attn_scores = torch.cat(attn_scores, dim=-1) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # combine V (wrist stream があれば 4 本目追加)
        v_list = [v_tokens, v_adapter, v_task]
        if use_wrist:
            v_list.append(v_wrist)
        v_combined = torch.cat(v_list, dim=2)

        output = torch.matmul(attn_weights, v_combined)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        if self.use_proper_ffn:
            # Standard pre-LN Transformer block: x = x + attn(norm1(x))、x = x + ffn(norm2(x))
            x = x + output
            x = x + self.ffn_down(nn.functional.gelu(self.ffn_up(self.norm2(x))))
        else:
            # Legacy: residual 置換、貧弱 FFN (LN + Linear + ReLU)
            x = self.ffn(output + x)
        return x


class XVLAActionBlock(nn.Module):
    """X-VLA 流 self-attention-only transformer block (2026-04-24 #014).

    MLPResNetBlock_Pro の 3-stream (self / adapter / task) を **self-attn 1 本**に統一。
    action queries は concat'd pool (action + scene + wrist + proprio + soft_prompt) を
    単一 softmax で処理する。attention starvation 回避目的。

    Ref: X-VLA/models/transformer.py TransformerBlock (aux_visual_inputs は concat-to-pool 方式)
    """

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

        self.rope = RotaryPositionEmbedding(self.head_dim)

    def forward(self, x):
        """
        Args:
            x: (B, T_total, dim) where T_total = action + scene + wrist + proprio + soft_prompt
        Returns:
            (B, T_total, dim)
        """
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        def reshape_heads(t, B, L):
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        q = reshape_heads(q, B, T)
        k = reshape_heads(k, B, T)
        v = reshape_heads(v, B, T)

        # RoPE
        cos, sin = self.rope(seq_len=T, device=x.device, dtype=x.dtype)
        q, k = apply_rope(q, k, cos, sin)

        # Attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        # Residual + FFN
        x = self.ffn(output + x)
        return x
