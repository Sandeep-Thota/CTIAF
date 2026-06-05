"""CTIAF training loop (Algorithm 1).

Expects a list of (Conversation, y_intent, y_trust) built with
ctiaf.features.build_conversation. This is a minimal, single-GPU reference
trainer; batching here is per-conversation for clarity (each conversation is a
graph). See the paper, Section 4.5 and Table 12, for the hyperparameters.
"""
from __future__ import annotations
import argparse
import torch
from torch.optim import Adam

from src.ctiaf.model import CTIAF, combined_loss, Conversation
from src.ctiaf.modules import CPAE


def train(dataset, in_dim, epochs=30, lr=2e-4, weight_decay=1e-5, patience=5,
          device="cpu"):
    model = CTIAF(in_dim=in_dim).to(device)
    opt = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, bad = float("inf"), 0
    for epoch in range(epochs):
        model.train()
        total = 0.0
        buffer = []  # detached platform embeddings of earlier conversations
        for i, (conv, yi, yt) in enumerate(dataset):
            opt.zero_grad()
            pi, pt, e_p = model(conv)
            # contrastive alignment (Eq 14): current embedding (grad) vs. detached others
            if buffer:
                z = torch.stack(buffer[-7:] + [e_p])
                last = len(z) - 1
                pos = [(last, last - 1)]
                neg = [(last, 0)] if len(z) > 2 else []
                align = CPAE.alignment_loss(z, pos, neg)
            else:
                align = pi.new_tensor(0.0)
            loss = combined_loss(pi, torch.as_tensor(float(yi)),
                                 pt, torch.as_tensor(float(yt)), align, model)
            loss.backward(); opt.step()
            total += float(loss)
            buffer.append(e_p.detach())
        avg = total / max(len(dataset), 1)
        print(f"epoch {epoch:02d}  loss {avg:.4f}")
        if avg < best - 1e-4:
            best, bad = avg, 0
            torch.save(model.state_dict(), "ctiaf_best.pt")
        else:
            bad += 1
            if bad >= patience:
                print(f"early stopping at epoch {epoch}")
                break
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CTIAF reference trainer")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()
    print("Build a dataset with ctiaf.features.build_conversation and pass it to "
          "train(). See examples/smoke_test.py for a minimal runnable example.")
