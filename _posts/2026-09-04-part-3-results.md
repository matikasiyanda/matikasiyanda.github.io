---
title: "Part 3 — Results, and the transfer test"
date: 2026-09-04
permalink: /blog/agent-rl/part-3-results/
series: "Training a document-search agent with GRPO"
series_url: /blog/agent-rl/
part: 3
description: "Two runs, the slice tables, and an insurance-trained agent pointed at Rust and JavaScript books."
image: /assets/agent-rl/v2_comparison.png
tags: [rl, grpo, retrieval, agents]
math: true
mermaid: false
---

*Last of three posts on building an agentic retrieval model on one RTX 4090:*

- [Part 1 — The data](/blog/agent-rl/part-1-data/): corpus, questions, verification, the
  failure→data map.
- [Part 2 — The environment and the RL](/blog/agent-rl/part-2-rl/): tools, token buffer,
  reward, GRPO.
- **Part 3 — Results** (you are here).
- [Glossary](#glossary) · [References](#references).

---

## v1: a working agent, and three honest failures

![v1 training curves](/assets/agent-rl/v1_training_curves.png)

***Figure 1.*** *v1 training curves over 200 steps. The dotted lines mark the context-schedule transitions; the run also survived a fragmentation OOM and a restart at step 95.*

200 steps, Qwen3-1.7B + LoRA (Figure 1). It beat everything it was measured against:

| policy | NDCG (held-out, n=203) | precision |
|---|---|---|
| BM25 top-10 | 0.232 | 0.05 |
| ColBERT top-10 | 0.338 | 0.07 |
| hybrid top-10 | 0.317 | 0.06 |
| SFT agent (no RL) | 0.229 | 0.21 |
| **v1 RL agent** | **0.458** | **0.333** |

On a *third* question set (fresh generator, fresh seed, zero text overlap) it
scored **0.404** — so the number wasn't a generator artifact.

The behavioural story was equally clear. RL taught **persistence** (1.5 → 3.3
turns; the SFT policy reported after one search wave), **reformulation**
(situational questions went 0.000 → 0.281), and **aimed fan-out**
(conjunctive 0.058 → 0.402).

But the slice table refused to flatter:

| slice | v1 |
|---|---|
| chain | **0.245** — flat all run, *falling* through the phase built to help it |
| situational | 0.281 (vs 0.546 direct) |
| single | 0.332 — below ColBERT's 0.404 |

And operationally it was violent: a fragmentation OOM at step 95, a corrupted
merged checkpoint, five distinct memory bugs before the first clean step, and
a hedging drift I watched but couldn't diagnose from the metrics alone.

Every one of those became a v2 change — the data fixes in
[Part 1](/blog/agent-rl/part-1-data/), the mechanics fixes in [Part 2](/blog/agent-rl/part-2-rl/).

---

## v2: the run

![v2 training curves](/assets/agent-rl/v2_training_curves.png)

***Figure 2.*** *v2 training curves over 200 steps — no crashes, no restarts, no interventions. The bottom-right panel is the trainer-versus-sampler logprob gap, flat at 0.011 nats throughout.*

**200/200 steps. Zero crashes. Zero restarts. Zero interventions** (Figure 2).

| phase | reward | NDCG | over-report | abstain-on-answerable | logprob gap |
|---|---|---|---|---|---|
| steps 0–25 | 0.476 | 0.408 | 1.21 | 0.09 | 0.0108 |
| steps 26–50 | 0.507 | 0.434 | 1.49 | 0.05 | 0.0111 |
| steps 51–199 | 0.593 | **0.525** | 1.96 | 0.05 | 0.0108 |

Two things to notice in Figure 2.

**It started where v1 finished.** Step 0 sampled at NDCG 0.457 — v1's final
score — because the solve-rate band feeds only questions that carry gradient
and the SFT floor was higher. The 10×-corrected learning rate then did in 25
steps what v1 took ~85 steps to do.

**The bottom-right panel is a flat line at 0.011 nats.** That's the
trainer-versus-sampler logprob gap: the collapse mode that kills naive
multi-turn RL implementations, measured as absent for all 200 steps.

### The one wart, and why I left it alone

Over-reporting climbed to ~1.96 — past the 1.5 line that is normally a
kill-criterion. I diagnosed it live rather than reflexively tuning:

- It was **failure-correlated**: high-ratio steps were the low-NDCG batches
  (2.30 ratio at 0.258 NDCG) while good batches stayed tight (1.11 at 0.489).
  The model hedges longer reports on questions it is failing.
- Equivalence classes already excluded the confound that made v1's version of
  this metric partly fictitious.
- Those hedged trajectories carry **negative advantage by construction** — the
  metric was counting behaviour the gradient was already punishing.
- Failed out-of-bounds rollouts inflate the ratio arithmetically (a 3-document
  report against an empty target set contributes 3.0).

I set tripwires (abstention > 0.30, phase-NDCG regression, final precision
materially below the SFT baseline's 0.360) and let it run. Final precision:
**0.352** — 0.8 points below baseline against a 4.3-point NDCG gain. The
patience was correct.

---

## Held-out results

![v1 vs v2 by slice, and the transfer gate](/assets/agent-rl/v2_comparison.png)

***Figure 3.*** *Left: v1 versus v2 on the held-out set, by slice. Right: the zero-shot transfer test on Rust and JavaScript books — a corpus the agent never trained on.*

Figure 3 puts v1 and v2 side by side, and the transfer test beside the
retrievers it beat.

**Main eval, n = 296, generated by a different pipeline than training:**

| policy | NDCG | precision | recall |
|---|---|---|---|
| BM25 top-10 | 0.334 | 0.075 | 0.443 |
| ColBERT top-10 | 0.425 | 0.094 | 0.551 |
| hybrid top-10 | 0.431 | 0.094 | 0.551 |
| SFT agent (no RL) | 0.499 | 0.360 | 0.548 |
| **v2 RL agent** | **0.542** | 0.352 | 0.609 |
| v2 RL agent + RRF@4 | **0.637** | 0.298 | 0.730 |

The precision column is the quiet headline: **0.352 vs 0.094**. The retrievers
dump ten documents and hope; the agent commits to two or three. For anything
downstream — a synthesis model, a human adjuster — that's the difference
between a citation and a reading list.

**Unseen test split (n = 139, third generator): 0.575.**

### v1 → v2 by slice

| slice | v1 | v2 | change |
|---|---|---|---|
| all | 0.458 | **0.542** | +18% |
| single | 0.332 | **0.481** | +45% |
| conjunctive | 0.402 | **0.516** | +28% |
| **chain** | 0.245 | **0.469** | **+91%** |
| situational | 0.281 | **0.420** | +49% |
| oob (abstention) | 0.780 | 0.746 | −4% |
| cross-insurer comparison | unmeasured | 0.591 | — |

Chains (Figure 3, left) — the slice that never moved in v1, that consumed the most turns and
tokens and converted the least — nearly doubled, and hit **0.697** on the
unseen test split. Nothing in the training algorithm changed to do that. The
entity-anchored pairing from [Part 1](/blog/agent-rl/part-1-data/) did it: 21% verified
yield instead of 10%, 314 chains with load-bearing bridges instead of 241
mostly-decorative ones.

That is the whole thesis of this project in one row of a table. **The
bottleneck was the data, and the data was fixable because the failures were
readable.**


### Did it learn to switch modes? Partly.

The question taxonomy from [Part 1](/blog/agent-rl/part-1-data/) exists for exactly one
purpose: the plan states that *"conjunctive questions should converge to fewer
turns than chain questions. If both flatten to the same number, the model
learned one mode, not two."* Figure 4 is that test.

![mode switching by question type](/assets/agent-rl/p3_mode_switching.png)

***Figure 4.*** *Turns and fan-out by question type across all three evaluation sets. The abstention/answerable distinction is learned cleanly; the conjunctive/chain distinction is barely there.*

| question type | turns (main) | turns (test) | turns (transfer) | calls/turn (main) |
|---|---|---|---|---|
| single | 2.99 | 2.97 | 2.85 | 2.81 |
| conjunctive | 2.98 | 3.00 | 3.00 | 2.68 |
| chain | 3.05 | 3.00 | 3.00 | 2.23 |
| **oob** | **2.51** | **2.49** | **2.44** | **1.89** |

Read honestly, this is a split verdict:

- **The distinction it learned is answerable versus not.** Out-of-bounds
  questions finish half a turn earlier and with a third less fan-out, on all
  three sets. The model recognises "there is nothing here" and stops, which is
  the behaviour with the clearest reward signal behind it.
- **The distinction it did not learn is conjunctive versus chain.** 2.98 versus
  3.05 turns on the main eval is a difference of 0.07 — by the plan's own
  criterion, *both flattened to the same number*. The agent runs essentially
  one answerable strategy: fan out wide, read what looks right, report.

That is the same finding as the turn-structure analysis in
[Part 2 §6.5](/blog/agent-rl/part-2-rl/) arriving from a different direction, and it is
consistent with chains being the weakest answerable slice (0.469). The one
place the strategies genuinely diverge is fan-out width: chains get 2.23 calls
per turn against single's 2.81, i.e. on the questions where a *second wave*
would help, the agent actually casts a *narrower* first net.

---

## The transfer test

Everything above could still be an elaborate way of memorising an insurance
corpus. So the last experiment removes the corpus.

Take the finished checkpoint — trained exclusively on South African insurance
documents — and point it, **with no further training**, at *The Rust
Programming Language* and six *You Don't Know JS* books. New corpus (1 692
chunks), new indexes, new questions from a separate generator run, same three
tools.

| policy | NDCG | precision | recall |
|---|---|---|---|
| BM25 top-10 | 0.333 | 0.070 | 0.419 |
| ColBERT top-10 | 0.380 | — | — |
| hybrid top-10 | 0.366 | — | — |
| SFT agent (pre-RL) | 0.760 | 0.609 | — |
| **v2 RL agent, zero-shot** | **0.781** | **0.606** | **0.815** |

That is the right-hand panel of Figure 3: three retriever bars around 0.33–0.38,
and the agent at 0.781.

**2.1× the best retriever on a corpus it had never seen.** Perfect abstention
(1.000) on unanswerable programming questions. And it was *more* efficient
there than at home — 2.73 turns and 117 tokens per question versus 2.89 and
134 — so it wasn't flailing at unfamiliar material, it was searching it.

Per-slice on the transfer corpus: single 0.597, conjunctive 0.658, chain
0.576, situational 0.802, cross-source comparison 0.767. RL *improved* on the
SFT starting point (0.760 → 0.781), so training sharpened the skill rather
than fitting the domain.

The two design decisions from Part 1 are what make this interpretable. The
agent outputs document IDs, so it cannot bluff from parametric knowledge about
Rust — it certainly *has* such knowledge, and that knowledge is unusable here.
The IDs are randomised per episode, so there was nothing to memorise even in
training. What transferred is the only thing that could transfer: reformulate,
fan out, read, verify, commit — or say it isn't there.

---

## Things that felt wrong and were correct

- **No parallelism reward.** Fan-out emerged from turn cost alone: 2.55
  calls/turn overall, 3.16 on situational questions — widest exactly where the
  vocabulary gap is hardest. A direct reward would have been farmed with junk
  queries.
- **No answer generation.** Document IDs make the agent composable as a
  subagent, memorisation-proof, and evaluable without an LLM judge in the loop.
- **~3 000 questions is plenty.** Per-episode ID obfuscation makes repetition
  safe; the 314 chains were each seen many times with different IDs.
- **Keeping the length bias in the advantage.** The "fix" is the bug.
- **Deliberately weak SFT.** It taught grammar and nothing else — and the
  gate (parse < 2%, calls/turn > 1.2) is what made RL's exploration possible.
- **Retrieval on CPU.** The GPU's job is rollouts.
- **Reading the failed questions by hand.** Every meaningful v2 gain came from
  that afternoon, not from the optimiser.

## What's next

- **4B.** Everything so far was bring-up at a size where bugs reproduce in
  minutes. Qwen3-4B-Instruct-2507 with FP8 is the next rung.
- **Privileged self-distillation (PBSD-style).** A teacher that sees the gold
  documents reweights the student's *own* per-turn advantages — turn-level
  credit assignment, which plain GRPO lacks. Gated on a stable curve, which v2
  now has.
- **Situational precision (0.193)** is the visible ceiling: the agent finds the
  right neighbourhood and hedges around it.
- **Abstention-aware fusion.** RRF@4 is worth +0.095 NDCG but currently lets
  one non-empty rollout override three abstentions.

---

## Appendix: everything in one table

| | v1 | v2 |
|---|---|---|
| Train questions (verified) | 2 236 | 3 104 |
| — chains | 241 (11%) | 314, of which 250 entity-anchored |
| Chain verification yield | ~10% | **21%** |
| Question keep rate | 62% | 71% |
| Learning rate | 1e-6 | **5e-6** |
| Sampling | pattern-weighted | **+ solve-rate band [0.15, 0.85]** (1 380 questions) |
| Scoring unit | single chunks | **equivalence classes** |
| Vocab masking / logprob-gap logging | none | yes (gap 0.011 nats, flat) |
| Crashes during the run | 1 OOM (+5 pre-run bugs) | **0** |
| Held-out NDCG | 0.458 | **0.542** |
| Held-out precision | 0.333 | 0.352 |
| Unseen-test NDCG | 0.404 | **0.575** |
| Chain slice | 0.245 | **0.469** (0.697 on test) |
| **Zero-shot transfer NDCG** | not run | **0.781** (best retriever there: 0.380) |

*Both runs, the data-generation code, the evaluation harnesses and the
failure→data map are self-contained in the project repositories — v1 frozen as
the baseline, v2 as the current line.*


---

## Glossary {#glossary}

<details markdown="1">
<summary>Plain definitions of every term used in the series. Click to expand.</summary>


### Retrieval and evaluation

**Corpus / chunk / document.** The *corpus* is the whole collection. A
*document* is one source file. A *chunk* is the retrievable unit — one section
(or part of a section) of a document, ~100–480 tokens here. The agent searches
and reports chunks, not whole documents.

**BM25 (lexical search).** The classic keyword-scoring function: a document
scores highly when it contains the query's rare words often, adjusted for
document length. It knows nothing about meaning — "car" and "vehicle" are
unrelated to it — which makes it excellent at exact names, numbers and
citations, and useless across a vocabulary gap.

**Embedding / dense retrieval.** Encode query and document into vectors and
rank by cosine similarity, so "vehicle" and "car" land near each other.
Standard dense retrieval squashes a whole passage into *one* vector.

**Late interaction / ColBERT.** A middle ground: encode *every token* into a
small vector, then score a query-document pair by, for each query token, taking
its best match among the document's token vectors and summing (called MaxSim).
Keeps word-level detail that a single vector loses, at the cost of storing many
vectors per chunk. The one I use is a 32M-parameter model running on CPU.

**PLAID.** The index structure that makes late-interaction search fast enough
to use (clustering + quantisation over all those token vectors).

**Hybrid search / RRF.** Running both retrievers and fusing their ranked lists.
*Reciprocal rank fusion* scores each document by $$\sum_r 1/(k + \text{rank}_r)$$
over the lists it appears in (I use $$k=60$$) — rank-based, so it needs no
score calibration between systems.

**Top-*k* / one-shot retrieval.** The baseline everything is compared against:
embed the question once, return the *k* best chunks, stop. No follow-up
queries, no reading, no way to abstain.

**Precision / recall.** Of the documents you returned, what fraction were
right (*precision*); of the right documents, what fraction did you return
(*recall*). A top-10 retriever has poor precision by construction — it always
returns ten things.

**DCG, IDCG, NDCG.** The metric this project optimises.
*Discounted Cumulative Gain* sums the relevance of returned items, discounting
by position, because a right answer at rank 1 is worth more than at rank 8:

$$\mathrm{DCG} = \sum_{i} \frac{g_i}{\log_2(i+1)}, \qquad g_i \in \{0, 1\}$$

*Ideal DCG* is the same sum for a perfect ordering (all targets first):

$$\mathrm{IDCG} = \sum_{i=1}^{|\text{targets}|} \frac{1}{\log_2(i+1)}$$

*Normalised DCG* is the ratio $$\mathrm{NDCG} = \mathrm{DCG}/\mathrm{IDCG}$$,
so 1.0 means "found everything, in the best order" and 0 means "found
nothing." Its logarithmic discount is why padding a report with extra
documents helps so little: rank 8 is worth $$1/\log_2 9 \approx 0.32$$ of
rank 1.

**pass@k.** Sample *k* independent attempts at a question; pass@k is the
fraction that succeed. Used here to measure how hard each question is for the
current policy (my *solve rate*).

**Zero-shot.** Evaluating a trained model on something it was never trained on
— here, a whole different corpus (programming books instead of insurance
documents) with no additional training.

---

### Reinforcement learning

**Policy ($$\pi_\theta$$).** The model itself, viewed as a thing that takes a
state and produces an action distribution. Here the state is the token buffer
so far and the action is the next token.

**Rollout / trajectory / episode.** One complete run of the agent on one
question: search → observation → read → observation → report. In this project
a trajectory is literally one array of token ids plus a mask saying which of
them the model produced.

**Turn.** One assistant message inside an episode (which may contain several
tool calls). Episodes here run 2–4 turns.

**Reward ($$R$$).** A single number scoring the finished episode — for me,
mostly NDCG over the reported documents, plus small format/budget/length
terms. *Sparse* because it arrives only at the end, not per token.

**Return / credit assignment.** The problem of deciding which of the many
decisions in an episode deserve the reward. Multi-turn agents make it hard: a
good first query may only pay off three turns later.

**Policy gradient.** The basic RL update for language models: increase the log
probability of tokens that led to better-than-expected outcomes, decrease it
for worse. Formally, ascend
$$\mathbb{E}[A \cdot \nabla_\theta \log \pi_\theta(a\mid s)]$$.

**Baseline and advantage ($$A$$).** Raw rewards are noisy, so you subtract a
reference value: the *advantage* is how much better this attempt was than
expected. Positive advantage → make those tokens more likely; negative →
less. PPO learns a value network to predict the baseline; GRPO does not.

**Group ($$G$$).** In GRPO, the several attempts at the *same* question that
are compared against each other (8 here).

**GRPO — Group Relative Policy Optimization.** The algorithm this project
uses. Instead of training a value network to estimate "expected reward from
this state", sample $$G$$ attempts at the same question and use their **mean
reward as the baseline**:

$$A_i \;=\; R_i - \frac{1}{G}\sum_{j=1}^{G} R_j$$

Cheaper (no critic to train or store), well-suited to problems where you can
sample many attempts, and the reason a 1.7B policy plus a serving engine fits
on one 24 GB card. Introduced in DeepSeekMath; popularised by DeepSeek-R1.
[Part 2 §6](/blog/agent-rl/part-2-rl/) covers how my version differs from the standard one.

**PPO — Proximal Policy Optimization.** The predecessor GRPO borrows its loss
shape from. It reuses a batch of rollouts for several gradient steps and, to
stop the policy moving too far, *clips* the ratio
$$r = \pi_\theta / \pi_{\theta_{\text{old}}}$$ between the new and old policies.
I take exactly one step per batch, so that ratio is always 1 and the clipping
does nothing — which is why my objective drops it.

**On-policy vs off-policy.** *On-policy* means you train on data the current
model just generated. If the sampler and the trainer disagree even slightly
(different precision, different kernels), you are silently a bit off-policy —
which is why I log the sampler/trainer logprob gap.

**KL divergence / KL penalty ($$\beta$$).** A measure of how far one
distribution has drifted from another. RLHF setups often add
$$-\beta\,\mathrm{KL}[\pi_\theta \Vert \pi_{\text{ref}}]$$ to keep the policy
near its starting point. I set $$\beta = 0$$.

**Zero-variance group.** A group where every attempt scores the same, so every
advantage is 0 and the group produces no gradient at all — pure wasted
compute. Avoiding these is what the solve-rate band is for.

**Logprob (log probability).** $$\log \pi_\theta(\text{token})$$ — the number
the policy gradient actually manipulates. Measured in *nats*; a gap of 0.011
nats between two computations of the same token means they agree to about 1%.

**Temperature / top-p.** Sampling controls. *Temperature* 1.0 means sample
from the model's raw distribution (more exploration, needed for RL); *top-p*
0.95 restricts sampling to the smallest set of tokens covering 95% of the
probability mass, cutting the pathological tail.

**Importance sampling (truncated IS).** A correction applied when the data was
generated by a slightly different policy than the one being updated — weight
each token by the ratio of the two probabilities, clipped for stability. My
fallback if the logprob gap ever grows; never needed.

---

### Training, adapters and serving

**Base model / instruct model.** The pretrained network (Qwen3-1.7B here) and
its instruction-tuned variant. "1.7B" = 1.7 billion parameters.

**SFT — supervised fine-tuning.** Training on demonstrations: given this
prompt, produce these exact tokens. Cheap and stable, but it can only imitate.
I use 400 traces of it purely to teach *tool syntax*, then stop.

**Teacher / trace.** A larger model (a local 27B) run through the same
environment to produce example episodes for SFT. A *trace* is one such
episode.

**RL vs SFT, in one line.** SFT teaches the model to copy a good trajectory;
RL teaches it to *find* one, by rewarding outcomes of its own attempts.

**LoRA — Low-Rank Adaptation.** Instead of updating all weights, freeze them
and learn a small low-rank correction $$\Delta W = BA$$ for chosen layers
(rank 32 here, ~1% of parameters). Makes training fit in memory, and the
result is a small adapter file you can swap in and out.

**Adapter hot-swap.** Because LoRA is a separate file, the serving engine can
load the freshly-updated adapter after each training step without reloading
the base model.

**Epoch / step / batch.** A *step* here is one full GRPO iteration: sample
rollouts for 4 questions × 8 attempts, compute rewards, take one gradient
update. 200 steps is the whole run.

**Gradient checkpointing.** A memory trick: discard intermediate activations
during the forward pass and recompute them during the backward pass. Trades
compute for memory; silently disabled if the model is left in eval mode (a bug
that cost me a day).

**KV cache.** The stored attention keys/values for the tokens generated so
far, so each new token doesn't reprocess the whole context. It is the dominant
memory cost during rollouts, and it is why long multi-turn episodes are
expensive.

**vLLM / colocation / sleep-wake.** vLLM is a fast serving engine (paged KV
cache, batching). *Colocation* means it shares the one GPU with the trainer;
I put it to *sleep* during the backward pass and *wake* it afterwards with
the new adapter.

**FP8 / bf16.** Numeric formats. `bf16` (16-bit) is what I train and serve
in; `fp8` (8-bit) halves the weight memory again and is the next step for a 4B
model.

**Context window / length schedule.** The maximum number of tokens an episode
may occupy (8k → 16k → 24k across training). Raising it late is a compute
allocation choice: cheap short episodes early, expensive long ones once the
policy can use them.

---

### Terms specific to this project

**Tool call.** A JSON block the model emits (`search`, `read`, `report`) that
the environment parses and executes, returning an observation.

**Fan-out.** Putting several queries in one `search` call rather than taking
several turns. Measured as *tool calls per turn*.

**Excerpt vs read.** `search` returns ~200-character excerpts; full text costs
a separate `read` call. Lets the agent triage many candidates cheaply.

**Breadcrumb.** The heading path of a chunk ("Santam: Personal Policy Wording
> Section 3 > Vehicle theft"), shown in every search result so the agent knows
*where* a hit sits in a document.

**TOC chunk.** A synthetic chunk per document containing its flattened heading
tree, indexed like any other chunk — a cheap way for the agent to orient in a
long document.

**ID obfuscation.** Remapping real chunk IDs to per-episode random tokens, so
the model cannot memorise "the answer is chunk a7f3c9" across training epochs.

**Equivalence class.** A set of near-duplicate chunks within a document
(cosine ≥ 0.9) treated as one unit for IDs, reporting and scoring, so
reporting an equally valid duplicate is not punished.

**Abstention / out-of-bounds (oob).** A question verified to be unanswerable
from the corpus; the correct behaviour is to search, find nothing, and
`report([])` — the empty report.

**Pattern tags: single / conjunctive / chain.** Question types by the search
policy they reward — one lookup; two independent facts (fan out in one turn);
a dependency where hop 2's search terms only exist in hop 1's result (search
sequentially). Never rewarded directly; used to check the model *switches*
modes.

**Situational phrasing / register.** Questions written as an incident in the
caller's own words ("a rock cracked my windscreen") rather than in document
vocabulary, optionally in an emotional register (frantic, angry, rambling,
post-trauma).

**Solve rate / band.** Per-question pass@8 with the current policy; training
samples only questions in the $$[0.15, 0.85]$$ band, where groups actually
produce gradient.

**Over-report ratio.** $$\text{mean}(|\text{reported}|)/\text{mean}(|\text{targets}|)$$
— the dashboard signal for padding reports. Sustained climb past 1.5 is a
kill-criterion.

**Token-in/token-out (TITO).** Feeding the model token *ids* and appending the
ids it returns verbatim, never re-rendering the conversation through a chat
template. The discipline that keeps multi-turn RL from collapsing.


</details>


---

## References {#references}

### The report this project is built on

- **[SID-1]** *SID-1 Technical Report*, SID AI, Dec 2025.
  <https://www.sid.ai/research/sid-1-technical-report>
  The source of: report-document-IDs-not-answers, hierarchical
  excerpt/read retrieval, per-episode ID obfuscation, strict token-in/token-out
  rollouts, keeping the length bias in the advantage, length scheduling, and
  the "don't train on public multi-hop datasets" argument. Almost every
  non-obvious decision in Parts 1–2 traces here.
- **[TP]** turbopuffer, *Training SID-1 to beat GPT-5 at search*, May 2026.
  <https://turbopuffer.com/blog/reinforcement-learning-sid-ai>

### RL algorithms

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

### Privileged / self-distillation (planned stage, not yet run)

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

### Retrieval

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

### Models, adapters, serving

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


