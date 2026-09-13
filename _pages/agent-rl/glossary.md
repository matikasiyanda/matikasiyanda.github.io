---
title: "Glossary"
permalink: /blog/agent-rl/glossary/
math: true
command: less ~/blog/agent-rl/glossary.md
---

<div class="prose" markdown="1">

Plain definitions of every term the three parts use.
[Part 1 — Data](/blog/agent-rl/part-1-data/) · [Part 2 — RL](/blog/agent-rl/part-2-rl/) ·
[Part 3 — Results](/blog/agent-rl/part-3-results/) · [References](/blog/agent-rl/references/).

---

## Retrieval and evaluation

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

## Reinforcement learning

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

## Training, adapters and serving

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

## Terms specific to this project

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


</div>
