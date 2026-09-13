---
title: "Work"
permalink: /work/
---

<div class="prose" markdown="1">

## Document-search agent, GRPO on one RTX 4090

<p class="meta work-meta">2026 · Qwen3-1.7B, LoRA, GRPO, vLLM</p>

Qwen3-1.7B + LoRA trained for 200 GRPO steps to *search* rather than
retrieve once: reformulate, chain hops, fan out, abstain. Then pointed,
zero-shot, at Rust and JavaScript books it had never seen.

<div class="stats">
  <div class="stat"><b>0.542</b><span>held-out NDCG (best one-shot retriever 0.431)</span></div>
  <div class="stat"><b>0.781</b><span>zero-shot NDCG on a new domain (retriever 0.380)</span></div>
  <div class="stat"><b>0.352</b><span>precision vs 0.094 for the retriever</span></div>
</div>

[Read the series](/blog/agent-rl/) · [data](/blog/agent-rl/part-1-data/) · [RL](/blog/agent-rl/part-2-rl/) · [results](/blog/agent-rl/part-3-results/)

## Product extraction from South African retail sites

<p class="meta work-meta">Nov 2022 – Sep 2023 · Selenium, Scrapy, spaCy, FLAN-T5-XL + LoRA, GPT-3.5</p>

The goal was a model that reads a store page's raw HTML and returns the
products on it, instead of a hand-written parser per store. I built scrapers
for Woolworths and Pick n Pay with Selenium, BeautifulSoup and Scrapy to get
the data, then tried three approaches.

The first was a spaCy NER model on a `roberta-base` transformer, trained on
1,378 Woolworths product cards (965 train, 413 test) to tag name and price
spans. The entity labels ended up as each product's own text rather than
`NAME` and `PRICE`, which gave 1,124 entity types, so the 0.84 F1 it reported
doesn't measure the task.

The second split the job in two with FLAN-T5-XL and LoRA adapters (rank 8 on
the attention q and v projections). One adapter decides whether a `<div>` is a
product card; it trained on 23,052 examples, balanced between product and
non-product divs from Woolworths Food and Pick n Pay. A second adapter
extracts the fields.

The third widened the net to more retailers, among them Dis-Chem, Bash and
Truworths. Of 258,546 scraped divs, 2,486 were product cards, and GPT-3.5
labelled those with item name, price, ID, promotion, promo price and URL as
JSON to make a training set.

Where it stopped: both adapters trained, but I never ran an end-to-end
evaluation, so there's no accuracy number I'd stand behind.

## Fine-tuning an OCR recogniser on synthetic text

<p class="meta work-meta">Aug 2022 · keras-ocr, TensorFlow, trdg, Colab</p>

The target was short strings mixing digits, letters and symbols. I wrote a
generator that drives [trdg](https://github.com/Belval/TextRecognitionDataGenerator)
in parallel across six categories (numbers, symbols, words and their pairings)
at font sizes 22 to 30, then fine-tuned the CRNN recogniser on the result with
a custom alphabet of digits, lowercase letters and punctuation, starting from
the pretrained backbone and using gamma-contrast augmentation.

On 7,200 held-out synthetic images it read 83.7% of strings exactly right, and
87.5% on the words-with-symbols subset. It still confused `l`
with `i` and `o` with `0`. Those numbers are exact-match on synthetic data;
it wasn't tested on real scans.

</div>
