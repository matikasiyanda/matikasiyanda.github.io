---
title: "Teaching a 14B model to find the fine print"
date: 2026-07-30
permalink: /blog/kb-search-agent-art/
description: "Insurance questions are answered by exclusions and conditions buried in the terms, not by the general cover wording. I trained Qwen3-14B with OpenPipe ART to search for them. It reached 0.769 on unseen questions, and RL wasn't what got it there."
tags: [rl, grpo, agents, retrieval, openpipe-art]
---

A customer types: *"I am trying not to panic. Does the vehicle warranty cover
electric vehicles or only hybrids? Please do not guess."*

The insurer's documents contain two answers. The warranty's general wording
covers mechanical breakdown and electrical failure, which sounds like yes. Its
exclusions section says electric vehicles are not covered, while hybrids are.
The exclusion is what decides a claim. An assistant that answers from the
general wording tells someone with an EV they're covered when they aren't, and
they find out at claim time.

That's the problem this project is about. Insurance questions are rarely
answered by the most relevant-sounding paragraph. They're answered by an
exclusion, a condition, a limit in an annexure, or by the fact that the
documents don't cover the question at all. I wanted a model that goes and finds
that part, cites it, and says "that isn't covered here" when it should.

Code: [github.com/matikasiyanda/kb-search-agent-art](https://github.com/matikasiyanda/kb-search-agent-art)

## Why ordinary retrieval gets this wrong

The usual approach is retrieval-augmented generation: embed the question, pull
the top few chunks by similarity, and have a model answer from them. On these
questions it fails in three ways.

It ranks by resemblance, not authority. "Does the warranty cover electric
vehicles" resembles the general cover wording and the product's marketing page
more than a terse exclusions list, so those come back first and the exclusion
may not come back at all.

It takes one shot. Asked whether an Uber vehicle can be insured, BM25 finds the
right policy document but not the clause about fare-paying passengers deeper
inside it, because the question never uses those words. Getting there takes a
second search after reading the first result.

It can't say no. Top-*k* always returns *k* chunks. Ask it to rewrite an angry
email, or for your live claim status, and it hands the model five insurance
paragraphs to improvise from.

A person handling these questions searches, skims, searches again using the
document's own words, reads the exclusions before answering, and knows when to
stop. The bet was that a model could learn that loop through reinforcement
learning, because "read the exclusions before you answer" is hard to enforce
with a prompt alone.

## What "right" meant

The knowledge base was one South African insurer's publicly released
brochures, plan guides, terms and annexures: 53 PDFs converted to markdown,
split on headings into sections with citation IDs, and indexed with Tantivy
BM25.

Every evaluation question carries an expected verdict, and the verdicts show
where the difficulty sits. Of the 147 unseen evaluation questions, only 2 were
plainly covered:

| expected verdict | questions |
|---|---|
| conditional cover | 50 |
| not found in the documents | 48 |
| excluded | 35 |
| needs clarification | 12 |
| covered | 2 |

Each question also records which sections must be read before answering, which
sources may not be the primary citation, and phrases the answer must never
contain. For the EV question:

```json
{
  "expected_verdict": "excluded",
  "required_read_citations": ["vehicle-warranty-terms-and-conditions::exclusions"],
  "forbidden_sources_as_primary": ["vehicle-warranty::what-makes-us-different"],
  "must_not_say": ["electric vehicles are covered", "full electric car is included",
                   "another insurer", "knowledge base"]
}
```

The product's marketing page ("what makes us different") is explicitly barred
as the main source. An answer that leans on it, or says EVs are covered, fails
however fluent it is.

## The agent and the training

The pattern comes from OpenPipe's ART·E example, an agent that searches an
email inbox. Here the model gets three tools: `search_kb` for a BM25 search,
`read_kb_section` to read a section in full, and `return_final_answer` with an
answer and its citations.

The model was `OpenPipe/Qwen3-14B-Instruct` with a rank-8 LoRA, trained 4-bit
through Unsloth while vLLM served rollouts, on a single rented 80 GB GPU on
RunPod. [ART](https://github.com/OpenPipe/ART) runs GRPO: for each question it
samples a group of attempts, scores each one, and nudges the model toward the
attempts that beat the group average. Pods bill by the minute, so each version
was one self-contained notebook plus a setup script.

Training questions were generated locally from the documents: 2,887 for the
first runs, then a deterministic generator with no LLM in the loop that
produced 11,909 (v2) and 15,400 (v5) templated questions with the same verdict,
citation and must-not-say fields. The run notes record two data mistakes along
the way. A set ordered easy to hard trained worse than a shuffled one, and a
set that was 80% unanswerable questions taught the model to decline everything.

## What happened

Accuracy here is gpt-4o-mini comparing each answer with the reference answer.

| run | reward | validation accuracy | test |
|---|---|---|---|
| run 3 | RULER only | 0.636 at step 10, 0.514 at step 60 | 0.650 on 20 |
| v2 | half RULER, half correctness | 0.707 → 0.709 | not saved |
| v3 | blended, plus `read_kb_section` | 0.836 at step 0, peak 0.881, ending 0.836 | 0.800 on 20 |
| v5 | correctness plus process rewards | 0.729 → 0.896 → 0.729 | 1.000 on 20, **0.769 on 147 unseen** |

Run 3 used RULER, ART's built-in reward, where an LLM ranks the attempts in a
group against each other without a reference answer. It's cheap, about $1.50
in judge calls for a 30-step run. The model learned what the judge liked, which
was sounding thorough, and accuracy fell as training went on.

v2 and v3 made half the reward a correctness check against a reference answer
the agent never sees, and the decline stopped. v3, which added the separate
read tool, is the best-behaved run. But look at step 0: the model scored 0.836
before any training, and its peak of 0.881 on 67 validation questions came
back down to where it started.

v5 added rewards for the process itself: searching before answering, reading
the required sections, citing them, and avoiding the must-not-say phrases. It
scored 1.000 on its 20-question test set. The same notebook then ran the final
checkpoint on the 147 unseen questions above.

![v5 accuracy, reward and forbidden-issue rate by step, with the unseen evaluation at step 41](/assets/art/v5_metrics.png)

| unseen eval, 147 questions | score |
|---|---|
| accuracy | 0.769 |
| citation score | 0.886 |
| behaviour score | 0.751 |
| forbidden-issue rate | 0.184 |
| searched before answering, read the required sections | 1.000 |

The model always searched and always read what it was meant to read, and still
said something forbidden in 18% of answers, against under 5% during training.
The process rewards couldn't have taught that search-and-read behaviour either:
search rate, read adherence, required-read recall and tool-trace score were
already 1.0 at step 0, so every attempt in a group scored the same on them and
GRPO had nothing to push against.

## Did it solve the problem?

Partly, and not because of the RL.

The capability that matters, finding the exclusion and reading it before
answering, was in the base 14B from the first step. On unseen questions the
trained model gets about three in four right and cites well (0.886), which is
useful. But it still makes a forbidden claim, like telling someone they're
covered, in roughly one answer in five, and that's the failure that hurts a
customer. The training numbers hid it: 1.000 on 20 questions written like the
training set says very little about 147 questions written to probe weak spots.

Three lessons went into the next attempt:

1. A strong base model leaves RL little room to show anything. Start with a
   model that can't yet do the task.
2. An LLM judge is a second model to game, and it bills per rollout. Score
   what can be checked mechanically, like which sections were cited.
3. Evaluate on hundreds of held-out questions built to be hard, from day one.

That attempt used Qwen3-1.7B on a single local RTX 4090, a from-scratch GRPO
loop, and a reward computed from reported document IDs with no judge. It's
written up in [three parts](/blog/agent-rl/).
