---
title: "Work"
permalink: /work/
command: "ls -l ~/work"
---

<div class="card" markdown="1">

## Document-search agent, GRPO on one RTX 4090

Qwen3-1.7B + LoRA trained for 200 GRPO steps to *search* rather than
retrieve once: reformulate, chain hops, fan out, abstain. Then pointed,
zero-shot, at Rust and JavaScript books it had never seen.

<div class="stats">
  <div class="stat"><b>0.542</b><span>held-out NDCG (best one-shot retriever 0.431)</span></div>
  <div class="stat"><b>0.781</b><span>zero-shot NDCG on a new domain (retriever 0.380)</span></div>
  <div class="stat"><b>0.352</b><span>precision vs 0.094 for the retriever</span></div>
</div>

[read the series](/blog/agent-rl/) · [data](/blog/agent-rl/part-1-data/) · [RL](/blog/agent-rl/part-2-rl/) · [results](/blog/agent-rl/part-3-results/)

</div>

<p class="todo"># TODO: more projects: name, one paragraph, links.</p>
