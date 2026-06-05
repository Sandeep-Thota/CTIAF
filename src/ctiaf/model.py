"""The CTIAF model: joins ISM, TARS, CPAE and DTG and predicts intent
satisfaction and trust (Eqs 19-21)."""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import TARS, CPAE, DTG


@dataclass
class Conversation:
    """One pre-processed conversation (see ctiaf.features for construction)."""
    node_feats: torch.Tensor      # [M, in_dim]  x_j  (Eq 16)
    adj: torch.Tensor             # [M, M] binary adjacency (E_seq + E_ref + E_topic)
    assistant_mask: torch.Tensor  # [M] bool
    profile_vec: torch.Tensor     # [7]   v_p (Eq 12)
    proxies: torch.Tensor         # [4]   trust proxies (Eqs 7-10)
    s_intent_adj: torch.Tensor    # scalar  ISM output (Eq 6)


class CTIAF(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, heads: int = 8,
                 gat_layers: int = 3, platform_dim: int = 64):
        super().__init__()
        self.dtg = DTG(in_dim, hidden=hidden, heads=heads, layers=gat_layers)
        self.cpae = CPAE(profile_dim=7, embed_dim=platform_dim)
        self.tars = TARS()
        fuse = self.dtg.out_dim + platform_dim + 1
        self.mlp_intent = nn.Sequential(nn.Linear(fuse, 128), nn.ReLU(), nn.Linear(128, 1))
        self.mlp_trust = nn.Sequential(nn.Linear(fuse, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, c: Conversation):
        g = self.dtg(c.node_feats, c.adj, c.assistant_mask)          # graph readout
        e_p = self.cpae(c.profile_vec)                               # Eq 13
        s_trust = self.tars(c.proxies)                              # Eq 11
        z_intent = torch.cat([g, e_p, c.s_intent_adj.view(1)])      # Eq 19
        z_trust = torch.cat([g, e_p, s_trust.view(1)])             # Eq 20
        return (self.mlp_intent(z_intent).squeeze(-1),
                self.mlp_trust(z_trust).squeeze(-1), e_p)


def combined_loss(pred_intent, y_intent, pred_trust, y_trust,
                  align_loss, model, weights=(1.0, 1.0, 0.5, 1e-4)):
    """L = w_i*BCE_intent + w_t*BCE_trust + w_a*L_align + w_r*||Theta||^2 (Eq 21)."""
    w_i, w_t, w_a, w_r = weights
    l_intent = F.binary_cross_entropy_with_logits(pred_intent, y_intent)
    l_trust = F.binary_cross_entropy_with_logits(pred_trust, y_trust)
    l2 = sum(p.pow(2).sum() for p in model.parameters())
    return w_i * l_intent + w_t * l_trust + w_a * align_loss + w_r * l2
