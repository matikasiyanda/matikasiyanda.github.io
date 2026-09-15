---
title: "Part 1 — The problem, and 145,000 clocks to study it with"
date: 2026-09-15
permalink: /blog/clocks/part-1-data/
series: "Lost in time"
series_url: /blog/clocks/
part: 1
description: "GPT-4o reads 8% of analogue clocks correctly. Before training anything: what does the task actually require, and how do you render enough clocks, diverse enough, with exact labels, to find out?"
tags: [vision, synthetic-data, pytorch]
math: true
image: /assets/clocks/same_time.png
---

*Lost in time, part 1 of 3. [Part 2](/blog/clocks/part-2-models/) trains
six encoders on the data built here. [Part 3](/blog/clocks/part-3-evaluation/)
looks at what they get wrong.*

## The problem

Show a multimodal language model a picture of a wall clock and ask for the
time. On the 62-clock ClockQA benchmark in *Lost in Time*
[[LostInTime]](#references), GPT-4o gets the exact time right 8% of the
time, Claude 3.5 Sonnet 6%, Gemini 2.0 23%, and Qwen2-VL-7B 0%. Roman
numerals and unusual hand shapes make it worse. Removing the second hand
doesn't help. The paper's reading is that the models can't detect the hands
and turn their angles into numbers, and nothing about a bigger language
model fixes that.

This is a strange failure. These models describe photographs, read
handwriting and answer questions about charts. A clock is a simpler image
than any of those. What it needs is not knowledge but measurement: find two
lines, measure their angles about a shared centre, and combine the two
readings. The image encoders inside those models were trained on
image-caption pairs, where "a clock on a wall" is a complete description and
the angle of the hands never matters. Nobody asked them to measure.

So this series asks the question the encoders were never asked. **Given a
128-pixel image of an analogue clock, output the hour and the minute.** Two
integers, 12 and 60 classes. Trained from random initialisation, which means
the network starts as pure noise, with no knowledge of edges, shapes or
clocks, and everything it ends up knowing it learned from these rendered
faces and nothing else. That's the opposite of the usual recipe, where you
start from a model pre-trained on millions of photographs. Starting from
nothing is slower and scores lower, but it means the result has one cause.

Three things I want to find out:

1. Can a small encoder learn to read a clock at all, from scratch, and how
   accurately? If a 3M-parameter network can do it, the failure above is not
   about capacity.
2. Does a convolutional network or a Vision Transformer learn it more
   easily at the same parameter count? The task is pure geometry, which is
   what convolutions are built for, and what attention has to learn.
3. When the models are wrong, how are they wrong? Off by one minute, or off
   by an hour? On which clocks?

The plan is the plain one. Render a large, deliberately varied set of clocks
with exact labels. Hold out one axis of variation, the fonts, to check that
the models read hands rather than memorise numerals. Train three ResNets and
three ViTs of matched sizes. Then look at the predictions, not just the
accuracy. This post is the data. Everything runs on one RTX 4090.

## What reading a clock requires

Two things make this harder than it looks, and both are about pixels rather
than semantics.

**The hour hand carries almost no information on its own.** On a real clock
it moves continuously, one numeral per hour, so at 4:58 it sits almost on
the 5. The renderer does the same. Here is one face at four times ([Figure 1.1](#fig-1-1)):

![The same clock at 4:57, 4:58, 5:00 and 5:02: the hour hand is visually identical, the label changes](/assets/clocks/hour_hand_coupling.png){: .no-invert}

**Figure 1.1.** The same clock at 4:57, 4:58, 5:00 and 5:02: the hour hand is visually identical, the label changes.
{: .figcap #fig-1-1}

Across those five minutes the hour hand rotates 2.5 degrees, about one pixel
at its tip. The label changes from 4 to 5 in the middle. The only way to get
the hour right is to read the minute hand first, notice it's near the 12,
and decide whether the hour hand is "just before" or "just after" the 5.
The two readings depend on each other, which is why Part 2 gives the hour
and minute separate output heads but the same encoder.

In symbols, with $$h$$ the hour, $$m$$ the minute, and angles in degrees
measured clockwise from 12:

$$\theta_m = 6m, \qquad \theta_h = 30h + \tfrac{1}{2}m,$$

so the hour hand advances half a degree per minute and the minute hand six.
Reading the hour means undoing the minute's contribution first:

$$h = \left\lfloor \frac{\theta_h - \theta_m / 12}{30} \right\rfloor \bmod 12.$$

Round $$\theta_h / 30$$ directly and 4:58 becomes 5:58. Whatever the model
learns internally, that subtraction is what it has to learn.

**One minute is three or four pixels.** The minute hand's tip sits at radius $$r$$
from the centre and moves along an arc of $$2\pi r / 60$$ per minute. The
renderer draws the face with radius between 30% and 47% of the image, and
the minute hand between 72% and 95% of that, so at 128 px

$$r \approx 0.85 \times 0.38 \times 128 \approx 41\ \text{px}, \qquad \frac{2\pi r}{60} \approx 4.3\ \text{px}$$

for a typical clock, and under 3 px for the smallest faces. Adjacent
minutes are adjacent classes, three or four pixels apart ([Figure 1.2](#fig-1-2)).

![10:37 and 10:38 on the same face, with the minute-hand tip enlarged: three pixels of difference](/assets/clocks/one_minute.png){: .no-invert}

**Figure 1.2.** 10:37 and 10:38 on the same face, with the minute-hand tip enlarged: three pixels of difference.
{: .figcap #fig-1-2}

The two enlarged panels are the whole difference between two classes, before
blur, noise and JPEG compression are applied. Keep that number in mind for
Part 2, where every model gets stuck at the same accuracy.

Everything above is about the hands. Numerals and tick marks, by contrast,
are decoration. A face with no
numerals is still readable if the model can find the centre and knows which
way is up. Whether the models use the numerals at all is a question the data
is built to answer.

## How a clock is drawn

The renderer is the part of this project I'd defend hardest, because
everything downstream inherits its choices. The tools are deliberately
ordinary. Every image comes from one function
of about 150 lines that draws a clock with Pillow, the standard Python
imaging library: `ImageDraw` for the ellipses, polygons and lines,
`ImageFont` for the numerals in TrueType and OpenType fonts found through
the system's `fc-list`. Rotation and the final downsample are Pillow's
`rotate` and `resize`; noise and brightness are three lines of NumPy; JPEG
artefacts come from encoding to a buffer and decoding again. The dataset
builder fans the work out across 12 cores with `multiprocessing.Pool` and
writes straight into memory-mapped NumPy arrays, so 145,000 clocks take
about three minutes and never sit in RAM all at once. Every sampled choice
is also written to a CSV alongside the labels, which is what Part 3 uses to
ask which kinds of clock are hard.

Before the full renderer, here is the whole idea in thirty lines that need
nothing but Pillow. Copy it into a file, run it, and you get the 512 px
clock in [Figure 1.3](#fig-1-3), reading 10:08.

<details markdown="1">
<summary>The thirty-line renderer (click to expand)</summary>

```python
"""A clock renderer in thirty lines, Pillow only. Saves clock_512.png"""
import math
from PIL import Image, ImageDraw, ImageFont

INK, FACE, BG = (30, 30, 40), (250, 250, 245), (245, 240, 225)


def draw_clock(hour, minute, size=512, ss=3, hands=INK):
    S = size * ss                            # draw at 3x, shrink later
    img = Image.new("RGB", (S, S), BG)
    d, c, R = ImageDraw.Draw(img), S / 2, S * 0.42

    def polar(r, deg):                       # angle clockwise from 12
        a = math.radians(deg - 90)
        return c + r * math.cos(a), c + r * math.sin(a)

    def disc(r, colour):
        d.ellipse([c - r, c - r, c + r, c + r], fill=colour)

    disc(R, INK); disc(R * 0.96, FACE)       # bezel, face
    for i in range(60):                      # ticks, longer every 5 min
        long = i % 5 == 0
        d.line([polar(R * 0.92, i * 6),
                polar(R * (0.82 if long else 0.88), i * 6)],
               fill=INK, width=ss * (4 if long else 2))
    font = ImageFont.truetype("DejaVuSans.ttf", int(R * 0.16))
    for h in range(1, 13):                   # numerals
        d.text(polar(R * 0.7, h * 30), str(h), fill=INK, font=font,
               anchor="mm")
    hour_angle = (hour % 12) * 30 + minute * 0.5    # moves with minute
    both = ((hour_angle, 0.55, 10), (minute * 6, 0.85, 6))
    for angle, length, width in both:
        d.line([polar(-R * 0.05, angle), polar(R * length, angle)],
               fill=hands, width=ss * width)
    disc(ss * 8, hands)                      # centre cap
    return img.resize((size, size), Image.LANCZOS)

if __name__ == "__main__":
    draw_clock(10, 8).save("clock_512.png")
```

</details>

![The clock drawn by the thirty-line snippet above](/assets/clocks/minimal_clock.png){: .no-invert}

**Figure 1.3.** The clock drawn by the thirty-line snippet above.
{: .figcap #fig-1-3}

Three things in it are the same three things the full renderer is built
around. It draws at three times the size and shrinks with a Lanczos filter,
because Pillow's lines are not anti-aliased. Every position comes from one
`polar` helper that takes an angle measured clockwise from 12. And the hour
hand's angle is `30 * hour + 0.5 * minute`, not `30 * hour`: it moves with
the minute, which is what makes 4:58 look like 5:00.

The full renderer in the repo does the same, then samples every choice the
snippet hard-codes. One call draws one clock at any size and hands back the
image ([Figure 1.4](#fig-1-4)) and a record of every choice it made:

```python
import random
from clockcheck.render_analog import render_analog
from clockcheck import fonts as F

rng = random.Random(42)
img, spec = render_analog(rng, out_size=512, font_pool=F.train_fonts(),
                          hour=10, minute=8)
img.save("clock_512.png")
print(spec)
```

![One clock rendered at 512 px by the snippet above](/assets/clocks/snippet_clock.png){: .no-invert}

**Figure 1.4.** One clock rendered at 512 px by the snippet above.
{: .figcap #fig-1-4}

And what `print(spec)` says about it:

```
AnalogSpec(hour=10, minute=8, second=40, face='circle', numerals='roman4',
           font='LinLibertine_RI.otf', ticks='all', hand_style='tapered',
           show_second=True, rotate=3.82, cx=258.2, cy=285.1, radius=207.5)
```

Every field is a choice the renderer made from the lists below: a circular
face, Roman numerals with IIII, all sixty tick marks, tapered hands, a second
hand at 40 seconds, a 3.8 degree rotation, and the centre and radius in
pixels. The dataset builder calls this function 145,000 times with different
seeds and writes each `spec` as one row of the metadata CSV.

The renderer draws at three times the output size and downsamples, so thin
hands and small numerals come out anti-aliased rather than jagged. Five
steps ([Figure 1.5](#fig-1-5)):

![One clock drawn in five stages: face and bezel, ticks, numerals, hands, then rotation, blur, noise and JPEG](/assets/clocks/stages.png){: .no-invert}

**Figure 1.5.** One clock drawn in five stages: face and bezel, ticks, numerals, hands, then rotation, blur, noise and JPEG.
{: .figcap #fig-1-5}

At each step the renderer samples from a list of options. This is what it
varies, with the share of the training set each option got ([Figure 1.6](#fig-1-6)):

At each step the renderer samples from a list of options. Here is one
example of each, so the names used from here on mean something
([Figure 1.6](#fig-1-6)):

![One example clock for every option: six numeral styles, four face shapes, four hand styles, three tick styles, with and without a second hand, rotated and upright](/assets/clocks/options.png){: .no-invert}

**Figure 1.6.** One example of every option the renderer chooses between, grouped by attribute, rendered at 512 px.
{: .figcap #fig-1-6}

This is what it varies, with the share of the training set each option got
([Figure 1.7](#fig-1-7)):

![Share of training images with each numeral style, face shape, tick style, hand style, second hand and rotation](/assets/clocks/knobs.png)

**Figure 1.7.** Share of training images with each numeral style, face shape, tick style, hand style, second hand and rotation.
{: .figcap #fig-1-7}

Beyond those categorical choices, every continuous quantity is sampled too:
the face radius (30% to 47% of the image), the centre offset (up to 6% off
centre), hand lengths and widths, numeral size and radius, tick lengths, and
four independent colours (background, face, bezel, hands) under a contrast
constraint so the hands are visible against the face. About 15% of numeral
sets rotate with their position around the dial rather than staying
upright. Half the time the hour hand also creeps with the second hand, as
some real movements do.

The last step damages the clean render, because a model trained on clean
renders won't survive a photograph. Blur up to 1.2 px, gaussian noise up to
14 levels, brightness between 0.75 and 1.2, JPEG at quality 60 or 80, each
applied with its own probability, and a rotation of up to 12 degrees.

The rotation has a rule. A bare pair of hands rotated by 90 degrees is a
different time, so the whole-image rotation is applied only when the face
has numerals or tick marks that fix which way is up. Otherwise the label
would be ambiguous and the model would be trained on noise.

Here is what all of that produces for one fixed time ([Figure 1.8](#fig-1-8)):

![4:58 rendered sixteen ways](/assets/clocks/same_time.png){: .no-invert}

**Figure 1.8.** 4:58 rendered sixteen ways.
{: .figcap #fig-1-8}

Every clock above says 4:58. The model has to learn that the pink one, the
one with no numerals, the one rotated on a green square, and the blurred one
are the same answer.

## Decisions that shaped the renderer

The first version of the renderer was a hundred lines and produced clocks a
model could learn from in an afternoon. Most of the work after that was
finding the ways it was subtly wrong. Four decisions, each with a
before-and-after ([Figure 1.9](#fig-1-9)):

![Four design decisions: supersampling, the colour contrast rule, and when rotation is a legal augmentation](/assets/clocks/design_decisions.png){: .no-invert}

**Figure 1.9.** Four design decisions: supersampling, the colour contrast rule, and when rotation is a legal augmentation.
{: .figcap #fig-1-9}

**Draw at three times the size, then shrink.** Anti-aliasing is what makes
a diagonal line look smooth on a screen: the pixels along its edge are
drawn in intermediate shades rather than fully on or off. Pillow doesn't do
it. It draws a one-pixel line as a one-pixel line. At 128 px a minute hand drawn
directly is a staircase, and the model would learn the staircase. Rendering
at 384 px and downsampling with a Lanczos filter gives every edge a soft
profile, which is also what a camera does. The left pair above is the same
clock both ways.

**Colours are random, but the hands must be visible.** Four colours are
sampled per clock: background, face, bezel, hands. With four independent
random colours, one clock in several has hands that nearly match the face,
and the label for that image is unlearnable. The rule is that the hand and
numeral colours must differ from the face in luminance by at least 90 (of
255). The second pair shows what the rule prevents.

**Rotate the image only when the image says which way is up.** Rotating the
whole image is a good augmentation: real clocks are photographed at angles.
But a face with no numerals and no tick marks rotated by 90 degrees is a
different time. 10:08 becomes 1:23. Rotation is applied only when the face
has numerals or ticks, and by at most 12 degrees. That's why the second row
above has two pairs: one where rotation would change the label, and one
where the 12 pins it down.

**Filter the fonts by rendering them.** The machine has 725 font files
that claim to cover English. A file-name blocklist for other scripts,
symbols and emoji cuts that to 357. The rest go through a glyph check:
render every digit and the Roman numeral letters to a bitmap, and require
all ten digits to be non-empty, distinct from each other, and distinct from
the font's missing-glyph box, because a font with no "7" renders a tofu
rectangle that has a perfectly good bounding box. That leaves 250.

While writing this post I checked what the 107 rejected fonts actually
failed on. All of them failed the same probe: their missing-glyph
placeholder is empty rather than a box, and the check treats an empty probe
as a failure. Every one of them renders the digits fine. So the filter is
stricter than it needed to be and the font pool could have been 357. It
doesn't change any result, since 250 fonts is plenty and the held-out split
is drawn the same way, but it's the kind of thing you only find by looking.

Two smaller ones. Roman dials sometimes use IIII rather than IV, so both are
rendered. And numerals are upright 85% of the time but rotate with their
position on the dial for the rest, because both conventions exist on real
clocks.

The digital-clock renderer was the fifth decision, and it went the other
way: built, trained on, and dropped, because reading seven-segment digits
is character recognition, with none of the geometry that makes this
problem interesting.

## Fonts as the held-out axis

A test set is a set of images the model never trained on, used to check
that it learned something general rather than memorising its training
images. When the test set is drawn from the same distribution as the
training set, a good score tells you the model didn't overfit. It doesn't
tell you *what* the model learned. I
wanted one axis of variation the model could plausibly cheat on, held out
entirely.

Fonts are that axis. If a model learns to read the numeral nearest each hand
tip rather than the hand's angle, it will be thrown by a typeface it hasn't
seen. If it reads angles, the typeface is irrelevant ([Figure 1.10](#fig-1-10)).

![Fourteen training fonts and fourteen held-out fonts, each rendering "12 3", with the font file name under each](/assets/clocks/fonts.png){: .no-invert}

**Figure 1.10.** Fourteen training fonts and fourteen held-out fonts, each rendering "12 3", with the font file name under each.
{: .figcap #fig-1-10}

The 250 fonts that survive the filter are split by family directory, so all
the weights of one family land on the same side: 211 for training, 39 held
out. The held-out test set is rendered with the 39 only. Here is what that
looks like at the level of whole clocks, seen typefaces above and unseen
below ([Figure 1.11](#fig-1-11)):

![Eight test clocks rendered with training fonts above eight rendered with held-out fonts](/assets/clocks/seen_unseen.png){: .no-invert}

**Figure 1.11.** Eight test clocks rendered with training fonts above eight rendered with held-out fonts.
{: .figcap #fig-1-11}

To a person the two rows are the same kind of thing. To a model that has
learned to match numeral shapes, the bottom row is new. That difference is
what Part 2 measures.

## The dataset

| split | images | fonts | seed offset | purpose |
|---|---|---|---|---|
| train | 120,000 | 211 training fonts | 1 | training |
| val | 5,000 | training fonts | 2 | picking the best epoch during training |
| test | 10,000 | training fonts | 3 | accuracy on the training distribution |
| test, held-out fonts | 10,000 | 39 held-out fonts only | 4 | accuracy on typefaces never seen |

All four splits come from the same renderer with the same settings. The
only things that differ are the random seed, which makes every image
unique, and for the last split the pool of fonts the numerals are drawn
from. There is no stratification step, and none is needed: with 120,000
draws from the same distribution, every category lands at the share the
renderer's probabilities say it should, in every split. The plot below is
the check. Each group of four bars is one option; the bars are the four
splits; they match ([Figure 1.12](#fig-1-12)).

![Share of each numeral style, face shape, tick style, hand style, second hand and rotation in each of the four splits](/assets/clocks/splits.png)

**Figure 1.12.** Share of each numeral style, face shape, tick style, hand style, second hand and rotation in each of the four splits.
{: .figcap #fig-1-12}

The labels are uniform by construction. Each image draws its hour from 12
values and its minute from 60, independently, so all 720 possible times
appear, about 167 times each in the training set. The second, which is
never labelled, is uniform too; it matters later because the minute hand
sometimes moves with it. The rotation angle, when applied, is uniform on
±12 degrees, with the un-rotated images making up the other 54% ([Figure 1.13](#fig-1-13)).

![Training images per (hour, minute) pair, the distribution of seconds, and the distribution of rotation angles](/assets/clocks/labels.png)

**Figure 1.13.** Training images per (hour, minute) pair, the distribution of seconds, and the distribution of rotation angles.
{: .figcap #fig-1-13}

Fonts are sampled uniformly from whichever pool the split uses, so every
training font appears on about 570 images and every held-out font on
about 260. No font appears in both ([Figure 1.14](#fig-1-14)).

![Images per font in the training pool and in the held-out pool](/assets/clocks/fonts_usage.png)

**Figure 1.14.** Images per font in the training pool and in the held-out pool.
{: .figcap #fig-1-14}

None of the above is about the pixels. The renderer's colour, lighting and
damage choices aren't recorded in the metadata, so the last plot measures
them from the images themselves: the mean brightness of each image, its
contrast (the standard deviation of its pixels), and its average colour.
The training set and the held-out set sit on top of each other, which is
what "only the fonts differ" should look like. And there is no favourite
palette, no preferred brightness, and a wide spread of contrast, from
near-monochrome faces to hard black-on-white ([Figure 1.15](#fig-1-15)).

![Per-image brightness, contrast and mean colour for a 20,000-image training sample and the held-out test set](/assets/clocks/photometrics.png)

**Figure 1.15.** Per-image brightness, contrast and mean colour for a 20,000-image training sample and the held-out test set.
{: .figcap #fig-1-15}

The whole set is 128 px RGB, 6.7 GB as uint8, and takes about three
minutes to build on 12 cores. Here it is, finally, as clocks rather than
histograms, from the held-out split with its labels ([Figure 1.16](#fig-1-16)):

![24 held-out-font clocks with their labels](/assets/clocks/samples_heldout.png){: .no-invert}

**Figure 1.16.** 24 held-out-font clocks with their labels.
{: .figcap #fig-1-16}

One consequence of the size: the whole training split fits on the GPU at
once. Part 2 uses that to skip the dataloader entirely.

## What's deliberately left out

- Real photographs. Every image is synthetic. A test set of real clocks
  [[ClockWild]](#references) is the test that would matter most, and there
  isn't one yet.
- Resolution above 128 px. Three or four pixels per minute is tight, and
  Part 2 has a result that suggests it's too tight. Testing that means rendering at
  192 or 224 px, which hasn't been done.

[Part 2 — ResNets against ViTs](/blog/clocks/part-2-models/) trains six
models on this data.

## References

- **[LostInTime]** Saxena, Gema & Minervini, *Lost in Time: Clock and
  Calendar Understanding Challenges in Multimodal LLMs*, 2025.
  arXiv:2502.05092. Source of the ClockQA accuracies quoted above.
- **[ClockWild]** Yang, Xie & Zisserman, *It's About Time: Analog Clock
  Reading in the Wild*, CVPR 2022. arXiv:2111.09162. Synthetic-to-real clock
  reading with real test sets from COCO, Open Images and *The Clock*.
