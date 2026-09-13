---
title: "Training a 14B search agent with OpenPipe ART on RunPod"
date: 2026-07-30
permalink: /blog/kb-search-agent-art/
description: "Qwen3-14B learns to search an insurer's brochures with GRPO and an LLM judge. The judge got gamed, the base model was already good, and the numbers were smaller than they looked."
tags: [rl, grpo, agents, retrieval, openpipe-art]
---

Before the [1.7B search agent on one RTX 4090](/blog/agent-rl/), there was this:
the same idea, a model that answers insurance questions by searching documents
instead of guessing, built on [OpenPipe ART](https://github.com/OpenPipe/ART)
with a 14B model on rented GPUs. Most of what went into the later project came
from what went wrong here.

Code: [github.com/matikasiyanda/kb-search-agent-art](https://github.com/matikasiyanda/kb-search-agent-art)

## The setup

The knowledge base was one South African insurer's publicly released
brochures and policy documents: 53 PDFs converted to markdown. An index script
splits each file on its `#` to `###` headings, caps sections at 2,000
characters and loads them into a Tantivy BM25 index, with a citation ID per
section.

The agent pattern comes from ART's ART·E example, an email-search agent. The
model gets a question and a small set of tools:

- `search_kb(query)` runs a BM25 search and returns matching sections with their IDs.
- `read_kb_section(id)` returns the full section (added in v3).
- `return_final_answer(answer, citations)` ends the episode.

The model was `OpenPipe/Qwen3-14B-Instruct` with a rank-8 LoRA, trained 4-bit
through Unsloth while vLLM served rollouts, both managed by ART's
`LocalBackend` on a single 80 GB GPU. ART runs GRPO: for each question it
samples a group of trajectories, scores them, and pushes the model toward the
ones that scored above the group's average.

## The data

Every training question was generated locally from the documents. The first
set, `rl_3k`, had 2,887 scenarios. Later sets came from a deterministic
generator with no LLM in the loop: templated questions over KB sections, 11,909
scenarios in 11 categories for v2, and 15,400 in 17 categories for v5. The
generator splits train and test by source group and rejects scenarios whose cited IDs don't exist or
whose questions duplicate another.

Two data lessons from the run notes:

- **Sorted data.** An earlier set was ordered easy to hard. That can help
  supervised fine-tuning, but in RL the policy changes every step, and the
  hard tail undid what the easy head had taught. Shuffling fixed it.
- **Too many "not in KB" questions.** One set was 80% questions the documents
  couldn't answer. The model learned to decline everything. The notes cap that
  share at 30% afterwards.

## Runs

Pods bill by the minute, so each version was a self-contained notebook plus a
setup script, with keys pasted straight in rather than wired through secrets.
Accuracy below is judged by gpt-4o-mini against a reference answer.

| run | reward | lr | steps | validation | test (20 q) |
|---|---|---|---|---|---|
| run 3 | RULER only | 1e-5 | 60 | 0.636 at step 10 → 0.514 at step 60 | 0.650 |
| v2 | 0.5 RULER + 0.5 correctness | 5e-6 | 30 | 0.707 → 0.709 | not saved |
| v3 | blended, + `read_kb_section` | 5e-6 | 30 | 0.836 → 0.881 (step 15) → 0.836 | 0.800 |
| v5 | correctness + process rewards | 1.5e-6 | 40 | 0.729 → 0.896 (step 36) → 0.729 | 1.000; **0.769 on 147 unseen** |

### Run 3: the judge got gamed

RULER, ART's built-in reward, has an LLM rank the
trajectories in a group against each other, with no reference answer needed.
It's cheap, about $1.50 of judge calls for a 30-step run. With RULER as the
only reward, the model got better at looking thorough and worse at being right.
Validation accuracy peaked around step 10 and fell to 0.514 by step 60. The
RunPod notes record the same run as 64% → 40%; the saved notebook output is
the number above.

### v2 and v3: blend in correctness

v2 made half the reward a correctness check against a reference answer the
agent never sees, halved the learning rate and shuffled the data. The curve
stopped falling but didn't climb either.

v3 added `read_kb_section`, giving the model a separate step for reading a
section in full. This is the run I'd stand behind: validation peaked
at 0.881. But the model at step 0, before any RL, already scored 0.836, and
validation here was 67 scenarios. A 4.5-point bump that returns to 0.836 by the
last step is inside the noise.

### v5: 1.000 on twenty, 0.769 on the rest

v5 added process rewards for searching, reading, citing and avoiding forbidden
statements, and reported 1.000 on the 20-question test set. The same notebook
also ran the final checkpoint (step 41) on 147 unseen scenarios written to
probe weak spots: adversarial source collisions, conditional cover, questions
the documents don't answer.

![v5 accuracy, reward and forbidden-issue rate by step, with the unseen evaluation at step 41](/assets/art/v5_metrics.png)

| unseen eval, 147 scenarios | score |
|---|---|
| accuracy | 0.769 |
| citation score | 0.886 |
| behaviour score | 0.751 |
| forbidden-issue rate | 0.184 |
| search rate, required-read recall | 1.000 |

The unseen set is the honest number. Accuracy drops from a perfect score to
0.769, and the forbidden-issue rate jumps from under 0.05 in training and
validation by the end of the run to 0.184: roughly one answer in five says
something it shouldn't once the questions stop looking like the training set.

The chart shows why process rewards didn't help. Search rate, read adherence,
required-read recall and tool-trace score sat at 1.0 from step 0, so every
trajectory in a group scored the same on them and GRPO had nothing to learn
from. The notebook also mentions gold traces in rollouts,
which may have leaked answers into training.

After v5 came an SFT warm-up (v6) and multi-turn conversations on an SFT'd
Qwen2.5-7B (v7 to v8.2). None of those have saved results.

## What carried over

Put side by side, the next project differs from this one in three ways, each
matching a problem above:

1. **The base 14B was already good.** RL on top moved accuracy by amounts
   smaller than the evaluation noise. A model that starts weak has more room
   to show what RL teaches.
2. **An LLM judge is a second model to game.** It costs money per rollout and
   rewards style. The next project scores the IDs the agent reports against a
   gold set, with no judge at all.
3. **Twenty test questions can't separate runs.** v5 looked perfect on 20 and
   scored 0.769 on 147. The next project evaluates on hundreds of held-out
   questions, split by the same slices used in training.

The follow-up moved to Qwen3-1.7B on a single local RTX 4090, with a from-scratch
GRPO loop instead of ART. That's [the three-part series](/blog/agent-rl/).
