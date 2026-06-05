"""Self-contained multi-head Graph Attention layer (Velickovic et al., 2018).

Dense-adjacency formulation so the module has no dependency on torch-geometric;
this keeps the reference implementation installable with a plain `pip install torch`.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 8,
                 concat: bool = True, dropout: float = 0.1, alpha: float = 0.2):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.concat = concat
        self.W = nn.Parameter(torch.empty(heads, in_dim, out_dim))
        self.a_src = nn.Parameter(torch.empty(heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.leaky = nn.LeakyReLU(alpha)
        self.drop = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """x: [N, in_dim]; adj: [N, N] binary (1 where an edge j<-k exists)."""
        n = x.size(0)
        h = torch.einsum("ni,hio->hno", x, self.W)               # [H, N, out]
        e_src = (h * self.a_src.unsqueeze(1)).sum(-1)             # [H, N]
        e_dst = (h * self.a_dst.unsqueeze(1)).sum(-1)            # [H, N]
        e = self.leaky(e_src.unsqueeze(2) + e_dst.unsqueeze(1))   # [H, N, N]
        mask = adj.unsqueeze(0) > 0
        e = e.masked_fill(~mask, float("-inf"))
        att = torch.softmax(e, dim=-1)
        att = torch.nan_to_num(att)        # isolated nodes -> all -inf -> nan
        att = self.drop(att)
        out = torch.einsum("hnm,hmo->hno", att, h)               # [H, N, out]
        if self.concat:
            return out.permute(1, 0, 2).reshape(n, self.heads * self.out_dim)
        return out.mean(0)
