"""Turn-level feature extraction and Dialogue Trajectory Graph construction.

Builds the per-conversation tensors consumed by ctiaf.model.CTIAF, following
Section 3 (node features Eq 16; sequential / referential / topic edges).
"""
from __future__ import annotations
import re
from typing import List, Dict
import torch
import torch.nn.functional as F

from .modules import intent_satisfaction, trust_proxies
from .model import Conversation

_CITE = re.compile(r"\[\d+\]|\((?:https?://|source)[^)]*\)|https?://\S+")
_REASON = re.compile(r"<think>(.*?)</think>|(?:step\s*\d|first,|second,|therefore|because)", re.I | re.S)
_CODE = re.compile(r"```")


def count_citations(text: str) -> int:
    return len(_CITE.findall(text or ""))


def reasoning_length(text: str, meta: Dict | None = None) -> int:
    if meta and meta.get("reasoning"):
        return len(str(meta["reasoning"]).split())
    m = _REASON.findall(text or "")
    return sum(len(str(x).split()) for x in m)


def build_conversation(turns: List[Dict], encoder, platform_profile: torch.Tensor,
                       m_max: int = 50, theta_ref: float = 0.7) -> Conversation:
    """turns: list of {text, role ('user'|'assistant'), meta}. `encoder` maps a
    list of strings to a [M, d] tensor (e.g. SentenceTransformer.encode)."""
    texts = [t["text"] for t in turns]
    roles = torch.tensor([0 if t["role"] == "user" else 1 for t in turns])
    emb = torch.as_tensor(encoder(texts), dtype=torch.float32)         # E(t_j)
    M, d = emb.shape

    n_cite = torch.tensor([count_citations(t["text"]) for t in turns])
    r_len = torch.tensor([reasoning_length(t["text"], t.get("meta")) for t in turns])
    tok = torch.tensor([len((t["text"] or "").split()) for t in turns], dtype=torch.float32)

    # node features x_j = [E(t_j); role_onehot; f_cite; f_reason]   (Eq 16)
    role_oh = F.one_hot(roles, num_classes=2).float()
    fc = torch.clamp(n_cite.float() / 5.0, max=1.0).unsqueeze(1)
    fr = torch.clamp(r_len.float() / 200.0, max=1.0).unsqueeze(1)
    node_feats = torch.cat([emb, role_oh, fc, fr], dim=1)

    # edges: sequential, referential (cos > theta_ref), topic (same coarse cluster)
    adj = torch.zeros(M, M)
    for j in range(M - 1):
        adj[j + 1, j] = adj[j, j + 1] = 1.0                            # E_seq
    sim = F.cosine_similarity(emb.unsqueeze(1), emb.unsqueeze(0), dim=-1)
    ref = (sim > theta_ref).float()
    ref.fill_diagonal_(0)
    adj = torch.clamp(adj + ref, max=1.0)                              # + E_ref
    # E_topic: greedy single-link clustering on cosine similarity (proxy for BERTopic)
    topic = (sim > 0.55).float(); topic.fill_diagonal_(0)
    adj = torch.clamp(adj + topic, max=1.0)

    assistant_mask = roles == 1
    a_emb = emb[assistant_mask]

    proxies = trust_proxies(n_cite, r_len, M, m_max, a_emb)            # Eqs 7-10
    s_intent_adj = intent_satisfaction(emb, roles, tok)                # Eqs 1-6

    return Conversation(node_feats=node_feats, adj=adj,
                        assistant_mask=assistant_mask,
                        profile_vec=platform_profile.float(),
                        proxies=proxies, s_intent_adj=s_intent_adj)


def platform_profile(stats: Dict) -> torch.Tensor:
    """v_p (Eq 12): [mu_len, sigma_len, mu_turn, r_cite, r_reason, r_code, mu_ratio]."""
    keys = ["mu_len", "sigma_len", "mu_turn", "r_cite", "r_reason", "r_code", "mu_ratio"]
    return torch.tensor([float(stats.get(k, 0.0)) for k in keys])
