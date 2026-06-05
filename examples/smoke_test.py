"""Minimal runnable example: builds two synthetic conversations, runs a forward
and backward pass through CTIAF, and prints metrics. Uses a random "encoder" so
it runs without downloading any model.

    python examples/smoke_test.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.ctiaf.features import build_conversation, platform_profile
from src.ctiaf.model import CTIAF, combined_loss
from src.ctiaf.modules import CPAE
from evaluate import intent_metrics, trust_metrics

D = 384  # all-MiniLM-L6-v2 dimension


def fake_encoder(texts):
    rng = np.random.default_rng(abs(hash(" ".join(texts))) % (2**32))
    return rng.standard_normal((len(texts), D)).astype("float32")


def demo_conversation(p_stats):
    turns = [
        {"role": "user", "text": "How do I sort a list in Python?", "meta": {}},
        {"role": "assistant", "text": "Use sorted(). See the docs [1]. https://docs.python.org", "meta": {}},
        {"role": "user", "text": "What about sorting by a key?", "meta": {}},
        {"role": "assistant", "text": "First, define a key function. Therefore use sorted(lst, key=...).", "meta": {}},
    ]
    return build_conversation(turns, fake_encoder, platform_profile(p_stats))


def main():
    torch.manual_seed(0)
    stats = {"mu_len": 1.1, "sigma_len": 0.4, "mu_turn": 5.3, "r_cite": 0.0,
             "r_reason": 0.3, "r_code": 0.2, "mu_ratio": 8.0}
    data = [(demo_conversation(stats), 1, 1), (demo_conversation(stats), 0, 0)]
    in_dim = data[0][0].node_feats.size(1)
    model = CTIAF(in_dim=in_dim)
    print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")

    buffer, preds_i, preds_t = [], [], []
    for conv, yi, yt in data:
        pi, pt, e_p = model(conv)
        # cross-conversation alignment: previous embeddings detached, current keeps grad
        if buffer:
            z = torch.stack(buffer + [e_p])
            align = CPAE.alignment_loss(z, [(len(z) - 1, len(z) - 2)], [])
        else:
            align = pi.new_tensor(0.0)
        loss = combined_loss(pi, torch.tensor(float(yi)), pt, torch.tensor(float(yt)), align, model)
        loss.backward()
        buffer.append(e_p.detach())
        preds_i.append(torch.sigmoid(pi).item()); preds_t.append(torch.sigmoid(pt).item())
        print(f"  conv -> intent_logit {pi.item():+.3f}  trust_logit {pt.item():+.3f}  loss {loss.item():.3f}")

    print("OK: forward + backward pass succeeded.")
    print("intent metrics:", intent_metrics([1, 0], preds_i))
    print("trust  metrics:", trust_metrics([1.0, 0.0], preds_t))


if __name__ == "__main__":
    main()
