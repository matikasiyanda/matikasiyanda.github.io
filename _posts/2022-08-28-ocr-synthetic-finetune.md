---
title: "Teaching an OCR model to read punctuation"
date: 2022-08-28
permalink: /blog/ocr-synthetic-finetune/
description: "keras-ocr's recogniser can't output a full stop, so it can't read a price. Adding punctuation meant retraining its output layer, and the labelled data for that came from a text-image generator."
tags: [ocr, keras, synthetic-data]
image: /assets/ocr/alphabet_problem.png
---

Look at a product page on a South African retail site and most of the text
that matters isn't words. It's `R 59.99`, `ID: 225163`, `50mm x 4.5m`. I wanted
an OCR model that could read that kind of text, and the one I started with
couldn't, for a reason that had nothing to do with image quality.

Code: [github.com/matikasiyanda/ocr-synthetic-finetune](https://github.com/matikasiyanda/ocr-synthetic-finetune)

## Why the pretrained model couldn't do it

keras-ocr [[keras-ocr]](#references) reads text in two stages. A detector
(CRAFT [[CRAFT]](#references)) finds boxes around words, and a recogniser (a
CRNN [[CRNN]](#references))
turns each cropped box into a string. The recogniser's last layer picks one
character from a fixed alphabet at each position, and the pretrained model's
alphabet is digits and lowercase letters. Nothing else.

The detector splits text at spaces, so `R 59.99` reaches the recogniser as two
crops, `R` and `59.99`. The second crop has a full stop in it, and the
recogniser has no output for a full stop. The best it can possibly return is
`5999`. The same goes for the colon in `ID:` and the decimal point in `4.5m`:

![Three price-page strings, their detector crops, and the best read each alphabet allows: the pretrained alphabet drops the full stop in 59.99, the colon in ID:, and the point in 4.5m](/assets/ocr/alphabet_problem.png){: .no-invert}

This is a ceiling, not a quality problem. No amount of cleaner images gets a
full stop out of a model that has no full stop to output; the alphabet has to
change.

Changing it has a cost. When you give the recogniser a different alphabet,
keras-ocr can't reuse the trained output layer, so it keeps the pretrained
backbone and trains a new output layer from scratch:

```python
alphabet = string.digits + string.ascii_letters + string.punctuation
recognizer = keras_ocr.recognition.Recognizer(alphabet=''.join(sorted(set(alphabet.lower()))))
# Provided alphabet does not match pretrained alphabet. Using backbone weights only.
```

That output layer needs thousands of labelled crops to train, and I didn't have
a pile of hand-labelled prices and product codes.

## Making the labels

trdg [[trdg]](#references) renders random
text as images and records exactly what it rendered, so every image comes with
a perfect label. I drove it from a script that generates six kinds of string,
from plain numbers like `194` to mixed runs like `bgy74j`:

| category | images |
|---|---|
| numbers | 2,000 |
| symbols | 2,000 |
| words | 6,000 |
| numbers + symbols | 2,000 |
| words + symbols | 6,000 |
| numbers + words | 6,000 |

The three word-based categories get three times as many images. Each run uses
one font size, and a later version of the script added skew between −15° and
15°. The script then renames every image with its category, merges trdg's label
files into one table, and zips the set for training on Colab. At font size 30
that's 24,000 labelled images.

This is what the generator produces, four images per category from a local run
at font size 24:

![Generated training images: rows of numbers, symbols, words, numbers with symbols, words with symbols, and numbers with words, in varied colours on speckled backgrounds](/assets/ocr/generated_samples.png){: .no-invert}

Text colour, background noise and blur vary from image to image. The
recogniser has to learn the characters, not one clean rendering of them.

## Training

The notebook starts from keras-ocr's own fine-tuning example and changes three
things: the data (the generated set instead of the example's Born-Digital
dataset [[BornDigital]](#references)), the alphabet (lowercase letters, digits and punctuation), and the
split. Labels are lowercased, then split 70/30 stratified by category so each
kind of string appears in both halves, leaving 16,800 images to train on and
7,200 to test.

Before an image reaches the model, keras-ocr resizes it to the recogniser's
fixed 200 × 31 input and pads the rest. Here is one training input from the
Colab run, labelled `^t+}i"]b` (labels are lowercased, so the capitals in the
image don't count against it):

![One training input: a short string of symbols and letters on a speckled background, padded with a flat block to 200 pixels wide](/assets/ocr/training_input.png){: .no-invert}

Training adds gamma-contrast augmentation on top, so the model sees the same
text lighter and darker, with early stopping once validation loss hasn't
improved for 10 epochs and the best checkpoint kept.

## How well it read

On the 7,200 held-out images, 83.7% of strings came out exactly right. A
typical correct read from the test set:

![A test image reading a8Ti0WB, which the model predicted as a8ti0wb, matching the label](/assets/ocr/prediction.png){: .no-invert}

Exact match is strict: `a8ti0wb` read as `a8tiowb` counts as wrong, even though six
of seven characters are correct. The notebook notes the errors that survived:
`l` read as `i`, and `o` read as `0`. Those are also the pairs that are hardest
for a person to tell apart in many fonts.

The notebook reports 87.5% on the words-with-symbols category alone. I'd leave
that number out. It comes from a second split of the data done separately from
the first, so some of those test images may have been in training.

## What it doesn't show

Every number here is synthetic text scored against synthetic text. A real
screenshot adds things the generator never produced: coloured backgrounds,
anti-aliased web fonts, a currency symbol next to a logo, and a detector that
has to crop the text before the recogniser sees it. The model was never
evaluated on real crops, so 83.7% says the training worked, not that it reads
product pages.

If I picked it up again, the first step would be fifty hand-labelled crops
from real pages, kept aside as the only test set that counts, with the
generator tuned until the synthetic images look like them.

## References

- **[keras-ocr]** F. Morales, *keras-ocr*: a packaged version of the CRAFT
  detector and a Keras CRNN recogniser. <https://github.com/faustomorales/keras-ocr>.
  Default recogniser alphabet: `string.digits + string.ascii_lowercase`.
- **[CRAFT]** Baek et al., *Character Region Awareness for Text Detection*,
  CVPR 2019. arXiv:1904.01941.
- **[CRNN]** Shi, Bai & Yao, *An End-to-End Trainable Neural Network for
  Image-based Sequence Recognition and Its Application to Scene Text
  Recognition*, IEEE TPAMI 39(11), 2017. arXiv:1507.05717.
- **[trdg]** E. Belval, *TextRecognitionDataGenerator*.
  <https://github.com/Belval/TextRecognitionDataGenerator>
- **[BornDigital]** Karatzas et al., *ICDAR 2013 Robust Reading Competition*,
  ICDAR 2013 (Challenge 1, "Born-Digital Images"). <https://rrc.cvc.uab.es/?ch=1>
