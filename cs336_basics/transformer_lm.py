import torch
from torch import nn
from einops import einsum, rearrange
from .transformer_block import Transformer_block
from .embedding import Embedding
from .utils import softmax
from .rmsnorm import RMSNorm
from .linear import Linear

class Transformer_LM(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            vocab_size: int,
            context_length: int, 
            num_layers: int,
            apply_rope: bool=True,
            theta: float | None=None,
            dtype: torch.dtype | None=None,
            device: torch.device | None=None
    ):
        super().__init__()
        self.d_model = d_model,
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.apply_rope = apply_rope
        self.theta = theta

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [Transformer_block(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                apply_rope=apply_rope,
                theta=theta,
                max_seq_len=context_length,
                dtype=dtype,
                device=device
            ) for _ in range(num_layers)]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token embedding
        x = self.token_embeddings(token_ids)
        # transformer block
        for block in self.layers:
            x = block(x)
        # layer norm
        norm_out = self.ln_final(x)
        # output embedding
        out_embedding = self.lm_head(norm_out)
        # softmax
        # result = softmax(out_embedding, dim=-1)
        return out_embedding