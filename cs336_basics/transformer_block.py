import torch
from torch import nn
from einops import einsum, rearrange
from .rmsnorm import RMSNorm
from .multihead_self_attention import Multihead_self_attention
from .swiglu import SwiGLU


class Transformer_block(nn.Module):
    def __init__(
            self, 
            d_model: int, 
            num_heads: int, 
            d_ff: int,
            apply_rope: bool=True,
            theta: float | None=None,
            max_seq_len: int | None=None,
            dtype: torch.dtype | None=None, 
            device: torch.device | None=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        # rmsnorm
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        # MHA
        self.attn = Multihead_self_attention(d_model, num_heads, apply_rope=apply_rope, theta=theta, max_seq_len=max_seq_len, dtype=dtype, device=device)
        # FFN
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None=None):
        # sub_layer_1
        # pre-norm
        norm1_output = self.ln1(x)
        # mha
        mha_output = self.attn(norm1_output, token_positions)
        # 残差连接
        y1 = x + mha_output
        # sub_layer_2
        norm2_output = self.ln2(y1)
        # ffn
        ffn_output = self.ffn(norm2_output)

        return ffn_output + y1