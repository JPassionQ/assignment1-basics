import torch
from torch import nn
from einops import einsum, rearrange
from .utils import scaled_dot_product_attention
from .rope import RoPE
from .linear import Linear

class Multihead_self_attention(nn.Module):
    def __init__(self,
                d_model: int,
                num_heads: int,
                apply_rope: bool = False,
                theta: float | None=None,
                max_seq_len: int | None=None,
                dtype: torch.dtype | None=None,
                device: torch.device | None=None):
        """
        因果语言模型多头自注意力的实现
        d_model: transformer block 输入的隐藏层维度
        num_heads: 多头自注意力中头的数量
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.apply_rope = apply_rope
        self.theta = theta
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if apply_rope:
            self.rope_layer = RoPE(theta, self.d_k, max_seq_len, device=device)
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None=None):
        # 计算 queries，keys，valus
        queries = einsum(x, self.q_proj.weight, "... seq_len d_model, hdk d_model -> ... seq_len hdk")
        queries = rearrange(queries, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)

        keys = einsum(x, self.k_proj.weight, "... seq_len d_model, hdk d_model -> ... seq_len hdk")
        keys = rearrange(keys, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)

        values = einsum(x, self.v_proj.weight, "... seq_len d_model, hdv d_model -> ... seq_len hdv")
        values = rearrange(values, "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads=self.num_heads)

        # 对 query 和 key 应用 RoPE 位置编码
        if self.apply_rope:
            queries = self.rope_layer(queries, token_positions)
            keys = self.rope_layer(keys, token_positions)
        
        # 生成掩码
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)).unsqueeze(-3)

        # 计算 注意力
        attn_output = scaled_dot_product_attention(queries, keys, values, mask=mask)

        # 拼接所有注意力头
        attn_output = rearrange(attn_output, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)")

        # 结果
        output = einsum(attn_output, self.output_proj.weight, "... seq_len hdv, d_model hdv -> ... seq_len d_model")

        return output