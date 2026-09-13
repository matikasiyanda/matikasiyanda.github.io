---
title: "Training a document-search agent with GRPO on one RTX 4090"
permalink: /blog/agent-rl/
math: true
---

<div class="prose" markdown="1">

A three-part write-up. Read in order:

1. **[Part 1 — The data is the model](/blog/agent-rl/part-1-data/)**
   What the data literally is (start at §0 if you want to replicate this on
   your own documents), why one-shot RAG fails, structure-preserving chunking,
   equivalence classes,
   synthetic question generation and verification, and the failure→data map
   (entity-anchored chains, deep-doc seeding, hard abstention, retrievability
   filtering, solve-rate banding).

2. **[Part 2 — The environment and the RL](/blog/agent-rl/part-2-rl/)**
   The three-tool surface, per-episode ID obfuscation, the token-in/token-out
   buffer rule, the reward equations, the multi-turn rollout written formally,
   and **how my objective differs from textbook GRPO** (no std normalisation,
   no ratio/clipping, β=0, kept length bias, masked multi-turn buffers) — plus
   vLLM/trainer colocation, chunked selective logits, vocab masking,
   logprob-gap instrumentation, and kill criteria.

3. **[Part 3 — Results and the transfer test](/blog/agent-rl/part-3-results/)**
   Two runs, the slice tables, and what happened when an insurance-trained
   agent was pointed at Rust and JavaScript books it had never seen.

[Glossary](/blog/agent-rl/glossary/) — plain definitions of NDCG, GRPO, LoRA, rollouts,
advantage, TITO and the rest.

[References](/blog/agent-rl/references/) — SID-1, GRPO/PPO, DAPO, Dr. GRPO, TITO,
ColBERT/PLAID, LoRA and the rest.

Figures: `v1_training_curves.png`, `v2_training_curves.png`, `v2_comparison.png`.

**Headline:** Qwen3-1.7B + LoRA, 200 GRPO steps — held-out NDCG 0.542 (best
one-shot retriever 0.431, precision 0.352 vs 0.094), unseen-test 0.575, and
**0.781 zero-shot on a different-domain corpus where the best retriever scores
0.380**.


</div>
