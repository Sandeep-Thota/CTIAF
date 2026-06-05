# CTIAF — Cross-Platform Trust and Intent Alignment Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Dataset: ShareChat](https://img.shields.io/badge/dataset-ShareChat-orange.svg)](https://huggingface.co/datasets/tucnguyen/ShareChat)

This repository accompanies the paper **"A Scalable Graph Neural Framework for
Cross-Platform Trust and Intent Modeling in Large Language Model Systems"**
(submitted to *Computers and Electrical Engineering*). It hosts the experimental
**results, figures, and documentation** for the Cross-Platform Trust and Intent
Alignment Framework (CTIAF).

## Overview

CTIAF is a data-centric framework for evaluating Large Language Model (LLM)
behaviour using real-world, multi-platform conversational data rather than
static benchmarks. It integrates four modules:

| Module | Role |
|--------|------|
| **ISM** — Intent Satisfaction Module | Models conversational completeness from semantic coverage and follow-up signals |
| **TARS** — Trust-Aware Response Scorer | Scores reliability from observable proxies: citations, reasoning traces, interaction depth, response consistency |
| **CPAE** — Cross-Platform Alignment Encoder | Learns platform-specific behavioural embeddings via contrastive learning |
| **DTG** — Dialogue Trajectory Graph | Models multi-turn interactions with a Graph Attention Network over sequential, referential and topic edges |

## Dataset

Experiments use the public **ShareChat** dataset — 142,808 conversations and
660,293 turns across five LLM platforms (ChatGPT, Claude, Gemini, Grok,
Perplexity) and 101 languages.

- Dataset: https://huggingface.co/datasets/tucnguyen/ShareChat (Apache 2.0)

See [`data/README.md`](data/README.md) for the partition / split accounting used
in the paper.

## Headline results

| Task | Metric | Best baseline | CTIAF |
|------|--------|---------------|-------|
| Intent satisfaction | F1 | 0.779 (ConvGAT) | **0.814** |
| Intent satisfaction | AUC-ROC | 0.841 (ConvGAT) | **0.879** |
| Trust estimation | MAE ↓ | 0.151 (ConvGAT) | **0.065** |
| Trust estimation | AUC-ROC | 0.821 (ConvGAT) | **0.927** |
| Trust estimation | Spearman ρ (vs. human labels) | 0.556 (ConvGAT) | **0.575** |
| Hallucination detection | F1 | 0.623 (G-Eval) | **0.730** |

All differences over the baselines are significant at *p* < 0.001 (pairwise
Wilcoxon signed-rank over 10 repeated splits; Cohen's *d* = 0.82–1.94). See
[`results/`](results/).

## Repository structure

```
src/ctiaf/       Reference implementation
  layers.py      Multi-head Graph Attention layer
  modules.py     ISM, TARS, CPAE, DTG  (Eqs 1-18)
  model.py       CTIAF model + combined loss  (Eqs 19-21)
  features.py    Turn features + Dialogue Trajectory Graph construction
train.py         Training loop (Algorithm 1)
evaluate.py      Evaluation metrics (Section 4.6)
examples/
  smoke_test.py  Minimal runnable forward/backward example
results/
  tables/        CSV exports of every results table (Tables 2-5, 13-16)
  figures/       Result figures (intent/trust comparison, per-platform, ablation, sensitivity)
data/
  README.md      Dataset reference and the corpus / split / subset accounting
```

## Usage

```bash
pip install -r requirements.txt
python examples/smoke_test.py          # runs a forward + backward pass on synthetic data
```

Build conversations from your data and train:

```python
from src.ctiaf.features import build_conversation, platform_profile
from train import train

# `encoder(list[str]) -> [M, d]` e.g. SentenceTransformer("all-MiniLM-L6-v2").encode
conv = build_conversation(turns, encoder, platform_profile(platform_stats))
model = train([(conv, y_intent, y_trust), ...], in_dim=conv.node_feats.size(1))
```

## Results files

| File | Paper table |
|------|-------------|
| `results/tables/table2_intent_satisfaction.csv` | Table 2 — intent satisfaction (all 10 baselines + CTIAF) |
| `results/tables/table3_trust_estimation.csv` | Table 3 — trust estimation |
| `results/tables/table4_per_platform_trust.csv` | Table 4 — per-platform trust (human-annotated subset) |
| `results/tables/table5_ablation.csv` | Table 5 — module ablation |
| `results/tables/table13_significance.csv` | Table 13 — Wilcoxon p / Cohen's d / 95% CI |
| `results/tables/table14_graph_topology.csv` | Table 14 — graph-topology comparison |
| `results/tables/table15_balanced_evaluation.csv` | Table 15 — balanced platform evaluation |
| `results/tables/table16_external_validation.csv` | Table 16 — temporal / language / domain validation |

## Reproducibility note

`src/ctiaf/` is a **clean reference implementation** of the CTIAF method exactly
as described in the paper (the equations in Section 3 and Algorithm 1). It is
provided for transparency and to make the architecture independently inspectable
and runnable. It is **not** a turnkey reproduction of the exact reported numbers:
those depend on the full ShareChat preprocessing pipeline, the human-annotated
label set, and trained checkpoints, which are released separately / available from
the authors on request. The metrics in `results/` are the values reported in the
manuscript.

## Citation

If you use this work, please cite:

```bibtex
@article{thota_ctiaf_2026,
  title   = {A Scalable Graph Neural Framework for Cross-Platform Trust and Intent
             Modeling in Large Language Model Systems},
  author  = {Thota, Sandeep Kumar and others},
  journal = {Computers and Electrical Engineering},
  year    = {2026},
  note    = {Under review}
}
```

## License

Code and documentation in this repository are released under the
[MIT License](LICENSE). The ShareChat dataset is distributed by its authors under
the Apache 2.0 license.
