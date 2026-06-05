# Data

CTIAF is trained and evaluated on the public **ShareChat** dataset:
https://huggingface.co/datasets/tucnguyen/ShareChat (Apache 2.0).

## Corpus, splits and evaluation subsets (as used in the paper)

| Partition | Size | Use |
|-----------|------|-----|
| Full corpus | 142,808 conversations | Source corpus (Table 1) |
| Pre-processed corpus | 128,493 | After removing sub-two-turn and duplicate conversations |
| Train / Val / Test (70/10/20) | 89,945 / 12,849 / 25,699 | Intent-classification metrics (Tables 2, 7, 8) |
| Human-validated gold set | ~5,000 | Stratified by platform; gold human intent + trust labels; seeds label propagation |
| Human-annotated trust-evaluation subset | 2,600 | Trust-correlation results (Tables 3, 4, Fig. 6): ChatGPT 1,629; Perplexity 417; Grok 273; Gemini 182; Claude 99 |
| Balanced-evaluation subset | 495 | 99 conversations per platform (Table 15) |

Trust labels were assigned by annotators who judged factual correctness and
source support directly (independent of the model's proxy features); see the
paper, Sections 4.2-4.2.2.
