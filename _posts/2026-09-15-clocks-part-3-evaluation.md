---
title: "Part 3 — What the models get wrong"
date: 2026-09-15
permalink: /blog/clocks/part-3-evaluation/
series: "Lost in time"
series_url: /blog/clocks/
part: 3
description: "Predictions on clocks in fonts never seen in training. Nearly every miss is one minute late, and the reason is a design choice in the renderer, not the models."
tags: [vision, vit, cnn, evaluation]
image: /assets/clocks/err_by_second.png
math: true
---

*Lost in time, part 3 of 3. [Part 1](/blog/clocks/part-1-data/) built the
data, [Part 2](/blog/clocks/part-2-models/) trained the models. This part
looks at what they predict.*

Accuracy tables say how often a model is right. They don't say what it does
when it's wrong, and on this task the wrong answers turned out to be the
interesting part. Two models get the close reading here: the best ResNet
(cnn_r34, 21M parameters, 30 epochs) and the modified ViT (vit_tiny_v2, 3.8M
parameters, stopped at epoch 86). The plain vit_tiny appears for contrast.
Everything below is on the 10,000-image test set rendered with the 39 fonts
no model saw in training, and every number is from one seed.

## The metrics

Three numbers are reported for every model. *Exact* means the hour and
the minute are both right. *Within one minute* means the hour is right and
the minute is at most one off. The *circular minute error* is how far off
the minute is, counted the short way round the dial, so 59 and 0 are one
apart. With predicted hour $$\hat h$$ and minute $$\hat m$$, and truth
$$(h, m)$$:

$$\text{exact} = [\hat h = h]\,[\hat m = m], \qquad
\text{within} \pm 1 = [\hat h = h]\,[\delta \le 1],$$

$$\delta = \min\big(|\hat m - m|,\ 60 - |\hat m - m|\big).$$

$$\delta$$ is the circular minute error, so a prediction of 59 for a true 0
counts as one minute off, not 59. The signed version
$$(\hat m - m + 30) \bmod 60 - 30$$ tells early from late.

| held-out fonts | exact | hour | minute | ±1 min | mean $$\delta$$ |
|---|---|---|---|---|---|
| cnn_r34 | 0.894 | 0.997 | 0.895 | 0.996 | 0.12 min |
| vit_tiny_v2 (86 ep) | 0.899 | 0.998 | 0.900 | 0.998 | 0.10 min |
| vit_tiny | 0.544 | 0.946 | 0.549 | 0.892 | 1.1 min |

Read across a row: the hour is almost never wrong, the minute is wrong one
time in ten, and nearly all of those are within one minute. Whatever the
models are doing, they are not confused about which clock they're looking
at.

## What the predictions look like

The most direct way to see a model read a clock is to draw what it read.
Below, each model's answer is drawn as a pair of hands over the clock it
was given: red for the ResNet, blue for the ViT, each with a black and
white halo so it shows on any background, and the labelled time as thin
yellow dashes on top. The clock itself is shown at a third of its colour
saturation so the hands stand out from it. Where all three coincide, the model
read the clock. The clocks are from the held-out set, so every typeface
here is one the models never saw. The geometry for the overlay comes from
re-rendering each test image from its seed, which reproduces it pixel for
pixel, and reading the centre and radius the renderer used.

The images are enlarged without smoothing, so the pixels are the pixels
the models saw, only paler.

![Predicted hands from both models, ResNet in red and ViT in blue, drawn over twelve held-out clocks in three rows: both right, one wrong, both wrong](/assets/clocks/overlay_heldout.png){: .no-invert}

In the top row the three sets of hands are on top of each other. In the
middle row one model is a minute off and the other isn't, and the two
readings differ by a few degrees at the tip. In the bottom row both models
agree with each other and disagree with the label, and the red and blue
minute hands sit just clockwise of the yellow dashes, one minute's worth. That is the
whole story of this post, and the rest of it is measuring that picture.

Another way to look at the same thing: take what each model answered and
draw it as a clean clock, next to the clock it was given. Red hands are
the ResNet's reading, blue hands the ViT's. If the model read the clock,
the drawn clock shows the same time as the unseen one; if it was a minute
late, the blue or red minute hand is one tick past where the real one
points.

![Six held-out clocks, each beside a clean clock drawn from the ResNet's reading in red and one from the ViT's reading in blue](/assets/clocks/readback_heldout.png){: .no-invert}

Eight more clocks each model read correctly, and eight it didn't, sampled
at random from the held-out set. The wrong readings are in red.

![ResNet-34 predictions on held-out fonts: eight right, eight wrong](/assets/clocks/preds_cnn_r34.png){: .no-invert}

![ViT tiny v2 predictions on held-out fonts: eight right, eight wrong](/assets/clocks/preds_vit_tiny_v2.png){: .no-invert}

And the same on the test set rendered with training-pool fonts, which is
the distribution the models were trained on:

![ResNet-34 predictions on the seen-font test set: eight right, eight wrong](/assets/clocks/preds_test_cnn_r34.png){: .no-invert}

![ViT tiny v2 predictions on the seen-font test set: eight right, eight wrong](/assets/clocks/preds_test_vit_tiny_v2.png){: .no-invert}

Look at the wrong rows. Every miss is one minute off, and most of them are
one minute late. The model reads 04:31 where the label says 04:32. Nothing
is off by an hour, nothing has confused the hands, nothing has been thrown
by a typeface.

## Every miss is one minute, and usually late

The signed error over all 10,000 held-out images, on a log scale so the
tail is visible:

![Distribution of predicted minus true minute on held-out fonts, log scale](/assets/clocks/minute_errors.png)

For the ResNet, 7.0% of predictions are one minute late and 3.4% one minute
early. Errors of two or more minutes are 0.16% of images, one in six
hundred. The ViT's tail is shorter still: 0.05%. The plain vit_tiny is the
contrast case, with 17% early and 20% late and a tail that reaches every
value on the axis.

Put plainly: when these models are wrong, they are wrong by one minute,
and more often they say the *next* minute than the previous one. Two things
about this distribution needed explaining. The errors are
almost all exactly one minute, and they're twice as often late as early.
A resolution limit would give the first but not the second: pixels don't
know which direction is later.

## The wall is in the renderer

Part 1 listed, among the renderer's choices, this one: "half the time the
hour hand also creeps with the second hand, as some real movements do." The
minute hand does the same. In the code, with probability one half,

$$\theta_m = 6m + \tfrac{s}{10},$$

so at $$s = 59$$ seconds the minute hand is 5.9 degrees past the minute
mark, a tenth of a degree short of the next minute. The label is still
$$m$$. The second hand itself is drawn on 45% of clocks and never labelled.

So a quarter of the training images (creep on, seconds past 30) show a
minute hand that is nearer the next minute than the labelled one. A model
that reads the angle honestly will round up on those, and be marked wrong.
Here is the error rate as a function of the second the renderer sampled:

![Late-by-one errors and exact accuracy as a function of the second, for both models](/assets/clocks/err_by_second.png)

Both models are at 96 to 97% exact for seconds 0 to 14 and fall to 70% for
seconds 45 to 59, on the same curve. In the last quarter of the minute,
23% of the ResNet's answers and 21% of the ViT's are one minute late. Two
architectures with nothing in common except the data draw the same line,
which is what you'd expect if the line belongs to the data.

The hour errors, rare as they are, are the same thing. The plot below has
one point per true minute wherever a model got the hour wrong at all:

![Hour error rate by true minute: the good models only fail at :00 and :59](/assets/clocks/hour_err_by_minute.png)

The ResNet's hour errors sit at minute 0 (6.5% of those images) and minute
59 (6.2%); the ViT's at 59 (4.9%) and 0 (3.9%). Everywhere else the rate
is below 1%, mostly zero. At 3:59 with 59 seconds and creep on, the minute
hand is on the 12 and the hour hand is on the 4. That image is, to the
pixel, 4:00. The model says 4:00 and the label says 3:59, and it goes down
as an hour error and a minute error at once. The plain vit_tiny, for
contrast, gets the hour wrong at every minute, because it can't read the
hands well enough for the boundary to matter.

The models have found a partial workaround on their own. When the second
hand is drawn, the late-by-one rate drops from 7.9% to 5.8% for the ResNet
and from 7.0% to 4.7% for the ViT. Nobody labelled the second hand. The
models learned that a second hand at 50 next to a minute hand just past the
mark means "not yet", because it lowered the loss.

This reframes the headline number. The 89 to 90% exact isn't a ceiling on
what the models can read. It's roughly what a perfect reader would score
against these labels. In the first quarter of the minute, where the creep
is under 1.5 degrees, both models are at 97%. The remaining 3% is the
combination of genuine resolution limits, the renderer's blur and noise,
and single-seed variance, and separating those is the next experiment. The
first thing to do is not to render at 224 px; it's to fix the labels, or
turn the creep off, and retrain.

I'd rather report this than hide it. The renderer's creep is a realistic
choice, since real clocks do it, and the bug isn't the creep, it's labelling
a crept clock with the floor of the minute. A person reading 3:59:59 on a
real clock says four o'clock too.

## Which clocks are hard

The renderer wrote every sampled choice to a CSV, so accuracy can be split
by what was drawn. The y axis starts at 0.80, because on these two models
everything lives between 0.85 and 0.92:

![Exact accuracy on held-out fonts by numeral style, face shape, hand style, tick marks, second hand and rotation](/assets/clocks/acc_by_attribute.png)

With 10,000 images split six ways, each bar rests on a few hundred to a
few thousand clocks, and a two-point gap is about the noise level. So most
of these bars say "no difference". Three of them say something.

**Faces with only a "12" are hardest for both models**, at 0.85 and 0.87
against 0.90 for full numerals, and faces with no tick marks are next. Both
models use the marks around the dial as a ruler for the minute hand. A face
with nothing on it forces the model to measure the angle against the centre
alone.

**Rotation hurts the ViT and barely touches the ResNet.** Upright faces:
0.91 for the ViT, 0.90 for the ResNet. Rotated ones: 0.88 and 0.89. The
plain vit_tiny, with learned position embeddings, is the extreme case: 0.66
upright and 0.41 rotated, a 25-point drop. Convolutions don't care where the
clock is; a transformer with position embeddings has to learn that a
rotated dial is the same dial, and the fixed sincos positions in v2 only
get it part of the way there.

**Outline hands are the one place the models disagree.** The ResNet's best
category (0.92) is the ViT's worst (0.88). An outlined hand is a thin
closed shape rather than a filled one, and a stride-1 convolutional stem
sees its edges at full resolution, where the ViT's stem has already
downsampled by four.

The held-out fonts, the thing the test set was built to probe, appear
nowhere in this list because they made no difference. Roman, arabic, none:
all the same. The models read hands.

## Where the two models disagree

On the held-out set, the ResNet and the ViT are both right on 83.9% of
clocks and both wrong on 4.7%. The ResNet alone is right on 5.4% and the
ViT alone on 5.9%. Either one right: 95.3%.

![Held-out clocks where both models are right, only the ResNet is right, only the ViT is right, and both are wrong](/assets/clocks/compare_heldout.png){: .no-invert}

The disagreements are all one-minute disagreements, in both directions,
and the "both wrong" row is mostly the renderer's creep again: both models
late by one on the same clock. That the two models are wrong on *different* clocks
6% of the time each, on a task where the labels are ambiguous a quarter of
the time, says their errors are mostly coin flips at the boundary rather
than systematic blind spots. Averaging the two heads would gain a few
points, and wouldn't mean anything.

For completeness, the worst misses by circular minute error for each
model, drawn the same way as the panels at the top of the post:

![The four largest minute errors on held-out fonts for each model, with both models' hands drawn over the clock](/assets/clocks/overlay_worst.png){: .no-invert}

These are the one-in-six-hundred cases, and they're a different kind of
error: the model swapped the hands. At 8:08 the hour hand sits at 244
degrees, which as a minute hand would read 41, and the ResNet said 8:41.
The hands in that image are nearly the same length. The 10:34 read as 6:53
is the same swap the other way. A person would need a second look too.

## What this changes

The question in Part 1 was which architecture learns clock geometry more
easily from scratch. The answer from Part 2 was: the ResNets straight away,
the ViTs only after the tokeniser was changed, and then a 3.8M ViT matches
a 21M ResNet. This part adds that both of them hit the same 90%, that the
90% is the labels and not the models, and that the two architectures'
remaining errors are the same errors on the same clocks.

Left to do, in order:

- **Fix the labels.** Either label the crept minute with the nearest
  minute, or turn creep off, and retrain cnn_small and vit_tiny_v2. The
  exact rate should jump to the mid-nineties for both. If it doesn't, the
  resolution question is back.
- **Finish the ViT sweep.** The plain vit_tiny at 100 epochs, and the four
  ablations, so that the tokeniser claim in Part 2 stands on more than one
  run.
- **The angle head.** Regressing $$(\sin\theta, \cos\theta)$$ per hand is
  the loss that would treat a crept clock as almost right rather than
  wrong. It's wired in and unimplemented.
- **Seeds.** Two runs of the same configuration differed by three points
  mid-training. Nothing in this series that's within two points of
  something else should be read as an ordering.
- **Real clocks.** Everything here is synthetic. A test set of photographed
  clocks [[ClockWild]](#references) is the experiment that decides whether
  any of this transfers.

The models themselves are available as PyTorch checkpoints; see the
[code repository](https://github.com/matikasiyanda/clock-check) for the
files and a loading snippet.

## References

- **[ClockWild]** Yang, Xie & Zisserman, *It's About Time: Analog Clock
  Reading in the Wild*, CVPR 2022. arXiv:2111.09162.
- **[LostInTime]** Saxena, Gema & Minervini, *Lost in Time: Clock and
  Calendar Understanding Challenges in Multimodal LLMs*, 2025.
  arXiv:2502.05092.
