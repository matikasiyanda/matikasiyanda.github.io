---
title: "Lost in time"
description: "Multimodal models can't read analogue clocks. Vision encoders trained from scratch on synthetic clock faces, CNNs against ViTs at matched size, then tested on real photographs. Three parts."
permalink: /blog/clocks/
---

<div class="prose" markdown="1">

![Sixty synthetic clock faces from the held-out test set](/assets/clocks/hero.png){: .no-invert}

Multimodal models can't read analogue clocks. This series trains vision
encoders from random initialisation on synthetic clock faces to find out what
the task actually needs, CNNs against Vision Transformers at matched parameter
counts, on one RTX 4090. Three parts, read in order:

1. **[Part 1 — The problem, and 145,000 clocks to study it with](/blog/clocks/part-1-data/)**
   Why frontier multimodal models read 8% of clocks correctly, what the
   task actually requires (two coupled angles, three or four pixels per minute),
   and how the synthetic clocks were designed: the renderer, what it
   varies, the decisions behind it, and fonts held out to catch cheating.

2. **[Part 2 — ResNets against ViTs](/blog/clocks/part-2-models/)**
   Six models from scratch. The CNNs read 89% of clocks exactly; plain ViTs
   underfit and get worse as they grow. Why the tokeniser is the problem,
   and the three changes that let a 3.8M ViT draw level with a 21M
   ResNet, given three times the training.

3. **[Part 3 — What the models get wrong](/blog/clocks/part-3-evaluation/)**
   Predictions on clocks in fonts never seen in training, the off-by-one
   wall every model hits, traced to a labelling rule and then fixed by
   retraining, every big miss shown one by one, and 3,228 real
   photographs, where the models manage 37% within a minute.

**Headline:** three ResNets, in 30 epochs, and one modified ViT, in 86,
all land at 88 to 90% exact on unseen fonts, and all of them stop in the
same place. That wall turned out to be a rounding rule in my own labels,
not the models: corrected, the smallest ResNet reads 95.3% in the same 21
minutes. On real photographs, all of them drop to about 37% within a
minute.

</div>
