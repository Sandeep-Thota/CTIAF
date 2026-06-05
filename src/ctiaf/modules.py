"""CTIAF modules: ISM, TARS, CPAE and DTG.

Reference implementation of the equations in Section 3 of the paper. Functions
that compute observable scores (Eqs 1-10, 12) are deterministic; learned pieces
(TARS aggregation Eq 11, CPAE encoder Eq 13, DTG message passing Eqs 16-18) are
nn.Modules.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GATLayer


# --------------------------------------------------------------------------- ISM
def intent_satisfaction(turn_emb: torch.Tensor, roles: torch.Tensor,
                        token_counts: torch.Tensor, tau: float = 200.0,
                        beta: float = 1.0, gamma: float = 0.8,
                        lambda1: float = 0.3) -> torch.Tensor:
    """Adjusted intent-satisfaction score s_intent^adj (Eqs 1-6).

    turn_emb: [M, d] turn embeddings; roles: [M] (0=user, 1=assistant);
    token_counts: [M] response token counts. Returns a scalar tensor in [0,1]-ish.
    """
    user_idx = (roles == 0).nonzero(as_tuple=True)[0]
    cov, weights = [], []
    M = turn_emb.size(0)
    for j in user_idx.tolist():
        if j + 1 >= M:
            continue
        h_u, h_a = turn_emb[j], turn_emb[j + 1]                       # Eq 1
        alpha = min(float(token_counts[j + 1]) / tau, 1.0)           # Eq 3
        rho = F.cosine_similarity(h_u, h_a, dim=0) * alpha           # Eq 2
        cov.append(rho)
        weights.append(math.exp(beta * (j + 1) / max(M, 1)))         # Eq 4
    if not cov:
        return turn_emb.new_tensor(0.0)
    cov = torch.stack(cov)
    w = turn_emb.new_tensor(weights)
    s_intent = (w * cov).sum() / (w.sum() + 1e-8)                    # Eq 4
    # follow-up penalty (Eqs 5-6)
    deltas = []
    for a, b in zip(user_idx[1:].tolist(), user_idx[:-1].tolist()):
        sim = F.cosine_similarity(turn_emb[a], turn_emb[b], dim=0)
        deltas.append(torch.clamp(sim - gamma, min=0.0))
    if deltas:
        s_intent = s_intent - lambda1 * torch.stack(deltas).mean()
    return s_intent


# -------------------------------------------------------------------------- TARS
def trust_proxies(n_cite: torch.Tensor, reason_len: torch.Tensor,
                  turn_count: int, m_max: int, assistant_emb: torch.Tensor,
                  eta: float = 5.0, kappa: float = 200.0) -> torch.Tensor:
    """The four trust proxy features (Eqs 7-10) -> tensor [4]."""
    f_cite = torch.clamp(n_cite.float().mean() / eta, max=1.0)                  # Eq 7
    has_reason = (reason_len > 0).float()
    f_reason = (has_reason * torch.clamp(reason_len.float() / kappa, max=1.0)).mean()  # Eq 8
    f_depth = math.log(1 + turn_count) / math.log(1 + max(m_max, 2))           # Eq 9
    if assistant_emb.size(0) >= 2:                                             # Eq 10
        sims = F.cosine_similarity(assistant_emb[1:], assistant_emb[:-1], dim=1)
        f_consist = 1.0 - (1.0 - sims).mean()
    else:
        f_consist = assistant_emb.new_tensor(1.0)
    return torch.stack([f_cite, f_reason,
                        assistant_emb.new_tensor(f_depth), f_consist])


class TARS(nn.Module):
    """Learned trust aggregation (Eq 11): s_trust = sigmoid(W_t . proxies + b_t)."""
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 1)

    def forward(self, proxies: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.lin(proxies)).squeeze(-1)


# -------------------------------------------------------------------------- CPAE
class CPAE(nn.Module):
    """Cross-Platform Alignment Encoder (Eqs 12-15)."""
    def __init__(self, profile_dim: int = 7, embed_dim: int = 64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(profile_dim, embed_dim), nn.ReLU())  # Eq 13

    def forward(self, profile_vec: torch.Tensor) -> torch.Tensor:
        return self.enc(profile_vec)

    @staticmethod
    def alignment_loss(z: torch.Tensor, pos_pairs, neg_pairs, margin: float = 1.0):
        """Contrastive alignment objective (Eq 14)."""
        loss = z.new_tensor(0.0)
        for i, k in pos_pairs:
            loss = loss + (z[i] - z[k]).pow(2).sum()
        for i, k in neg_pairs:
            d = (z[i] - z[k]).pow(2).sum()
            loss = loss + torch.clamp(margin - d, min=0.0)
        denom = max(len(pos_pairs) + len(neg_pairs), 1)
        return loss / denom


# --------------------------------------------------------------------------- DTG
class DTG(nn.Module):
    """Dialogue Trajectory Graph encoder: stacked GAT + hierarchical readout
    (Eqs 16-18)."""
    def __init__(self, in_dim: int, hidden: int = 256, heads: int = 8,
                 layers: int = 3, dropout: float = 0.1):
        super().__init__()
        per_head = hidden // heads
        self.gats = nn.ModuleList()
        d = in_dim
        for _ in range(layers - 1):
            self.gats.append(GATLayer(d, per_head, heads=heads, concat=True, dropout=dropout))
            d = per_head * heads
        self.gats.append(GATLayer(d, hidden, heads=heads, concat=False, dropout=dropout))
        self.att_pool = nn.Linear(hidden, 1)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor, adj: torch.Tensor,
                assistant_mask: torch.Tensor) -> torch.Tensor:
        for i, gat in enumerate(self.gats):
            x = gat(x, adj)
            if i < len(self.gats) - 1:
                x = F.elu(x)
        mean_pool = x.mean(dim=0)                                          # Eq 18
        if assistant_mask.any():
            xa = x[assistant_mask]
            w = torch.softmax(self.att_pool(xa), dim=0)
            att_pool = (w * xa).sum(0)
        else:
            att_pool = mean_pool
        return 0.5 * (mean_pool + att_pool)
