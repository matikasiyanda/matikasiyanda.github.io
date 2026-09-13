---
title: "References"
permalink: /blog/agent-rl/references/
math: true
---

<div class="prose" markdown="1">

Shared bibliography for the three-part write-up:
[Part 1 — Data](/blog/agent-rl/part-1-data/) · [Part 2 — RL](/blog/agent-rl/part-2-rl/) ·
[Part 3 — Results](/blog/agent-rl/part-3-results/).

## The report this project is built on

- **[SID-1]** *SID-1 Technical Report*, SID AI, Dec 2025.
  <https://www.sid.ai/research/sid-1-technical-report>
  The source of: report-document-IDs-not-answers, hierarchical
  excerpt/read retrieval, per-episode ID obfuscation, strict token-in/token-out
  rollouts, keeping the length bias in the advantage, length scheduling, and
  the "don't train on public multi-hop datasets" argument. Almost every
  non-obvious decision in Parts 1–2 traces here.
- **[TP]** turbopuffer, *Training SID-1 to beat GPT-5 at search*, May 2026.
  <https://turbopuffer.com/blog/reinforcement-learning-sid-ai>

## RL algorithms

- **[GRPO]** Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical
  Reasoning in Open Language Models*, 2024. arXiv:2402.03300 — introduces
  Group Relative Policy Optimization: group-sampled baselines, no value
  network.
- **[R1]** DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in
  LLMs via Reinforcement Learning*, 2025. arXiv:2501.12948.
- **[PPO]** Schulman et al., *Proximal Policy Optimization Algorithms*, 2017.
  arXiv:1707.06347 — the clipped surrogate GRPO inherits.
- **[DrGRPO]** Liu et al., *Understanding R1-Zero-Like Training: A Critical
  Perspective*, 2025 — proposes removing length/std normalisation biases.
  I deliberately **do not** adopt the length debiasing; see Part 2 §5.2 and
  [SID-1].
- **[DAPO]** Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System
  at Scale*, Mar 2025 — dynamic sampling that discards zero-variance groups.
  My solve-rate band is the offline version of this idea.
- **[TITO]** Hugging Face, *Agentic RL: Token-In, Token-Out Done Right*, May
  2026. <https://huggingface.co/blog/huggingface/tito> — the delta-append
  rollout loop and the chat-template prefix-preservation problem.
- **[IS]** Yao et al., *Your Efficient RL Framework Secretly Brings You
  Off-Policy RL Training*, 2025.
  <https://fengyao.notion.site/off-policy-rl> — inference/trainer logprob
  mismatch and truncated importance sampling; the reason I log the gap.

## Privileged / self-distillation (planned stage, not yet run)

- **[PBSD]** *Privileged Bayesian Self-Distillation for Long-Horizon Credit
  Assignment*, Jun 2026. arXiv:2606.09348 — advantage reweighting by an
  evidence score instead of a KL target.
- **[SDAR]** *Self-Distilled Agentic RL*, May 2026. arXiv:2605.15155.
- **[SkillSD]** *Skill-Conditioned Self-Distillation for Multi-turn LLM
  Agents*, Apr 2026. arXiv:2604.10674.
- **[OPID]** *On-Policy Skill Distillation for Agentic RL*, Jun 2026.
  arXiv:2606.26790.
- **[PathRouter]** *Gold-evidence teacher for agentic retrieval*, Jun 2026.
  arXiv:2606.16409 — masks KL on answer tokens; the reason my OPSD spec masks
  KL on `report` tokens.

## Retrieval

- **[NDCG]** Järvelin & Kekäläinen, *Cumulated Gain-Based Evaluation of IR
  Techniques*, ACM TOIS 20(4), 2002 — the metric my reward is built on.
- **[BM25]** Robertson & Zaragoza, *The Probabilistic Relevance Framework:
  BM25 and Beyond*, FnTIR 3(4), 2009.
- **[bm25s]** Lù, *BM25S: Orders of magnitude faster lexical search via
  eager sparse scoring*, 2024 — the lexical index I use.
- **[ColBERT]** Khattab & Zaharia, *ColBERT: Efficient and Effective Passage
  Search via Contextualized Late Interaction over BERT*, SIGIR 2020.
  arXiv:2004.12832.
- **[ColBERTv2]** Santhanam et al., *ColBERTv2: Effective and Efficient
  Retrieval via Lightweight Late Interaction*, NAACL 2022.
- **[PLAID]** Santhanam et al., *PLAID: An Efficient Engine for Late
  Interaction Retrieval*, CIKM 2022 — the index backend used via pylate.
- **[pylate]** LightOn, *PyLate* — late-interaction training/retrieval library.
  <https://github.com/lightonai/pylate>
- **[mxbai]** Mixedbread, *mxbai-edge-colbert-v0* (17M / 32M late-interaction
  retrievers for edge/CPU use), 2026 — my semantic retriever.
- **[RRF]** Cormack, Clarke & Buettcher, *Reciprocal Rank Fusion Outperforms
  Condorcet and Individual Rank Learning Methods*, SIGIR 2009 — hybrid mode
  and the 4× rollout fusion in Part 3.
- **[ARTE]** OpenPipe, *ART·E* — the email-search agent whose
  excerpt-then-read design SID-1 credits.

## Models, adapters, serving

- **[Qwen3]** Qwen Team, *Qwen3 Technical Report*, 2025 — the 1.7B policy and
  the 27B local generator/judge.
- **[LoRA]** Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*,
  2021. arXiv:2106.09685.
- **[LoRAReg]** Thinking Machines, *LoRA Without Regret*, Sep 2025 — LoRA's
  optimal LR sits ≈10× above full fine-tuning's, and low rank suffices for RL.
  The reason v2 moved from 1e-6 to 5e-6.
- **[vLLM]** Kwon et al., *Efficient Memory Management for Large Language Model
  Serving with PagedAttention*, SOSP 2023. arXiv:2309.06180.
- **[TRL]** Hugging Face TRL 1.9, Jul 2026 — multi-environment agentic GRPO;
  a supported alternative to the hand-rolled loop.
- **[Unsloth]** Unsloth, *FP8 Reinforcement Learning* docs — FP8 + weight
  sharing between vLLM and the trainer (planned for the 4B stage).


</div>
