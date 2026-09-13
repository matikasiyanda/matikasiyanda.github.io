---
title: "Teaching an OCR model to read symbols with synthetic data"
date: 2022-08-28
permalink: /blog/ocr-synthetic-finetune/
description: "Generating labelled text images with trdg and fine-tuning the keras-ocr recogniser on strings that mix digits, letters and punctuation."
tags: [ocr, keras, synthetic-data]
---

The target was short strings that mix digits, letters and punctuation, like
`&g;b-R`. Labelled real images of those are slow to collect and synthetic ones
are free, so I generated them and fine-tuned the text recogniser from
[keras-ocr](https://github.com/faustomorales/keras-ocr) on the result.

Code: [github.com/matikasiyanda/ocr-synthetic-finetune](https://github.com/matikasiyanda/ocr-synthetic-finetune)

## Generating the data

[trdg](https://github.com/Belval/TextRecognitionDataGenerator) renders random
strings as images and writes the labels next to them. The generator runs it
once per category, with each category a command-line recipe:

```python
uz = lambda x: {
    f"numbers_fontsize_{x}":       f"trdg -c {num} ... -rs -num -f {x}",
    f"symbols_fontsize_{x}":       f"trdg -c {num} ... -rs -sym -f {x}",
    f"words_fontsize_{x}":         f"trdg -c {3*num} ... -f {x}",
    f"numbers_symbols_fontsize_{x}": f"trdg -c {num} ... -rs -num -sym -f {x}",
    f"words_symbols_fontsize_{x}": f"trdg -c {3*num} ... -rs -let -sym -f {x}",
    f"numbers_words_fontsize_{x}": f"trdg -c {3*num} ... -rs -let -num -f {x}",
}
```

Six categories, with the word-based ones getting three times as many images,
black or grey text, fixed font sizes. A later version added skew from −15° to
15° and ran trdg with 8 threads. The script then renames every image with its
category, merges trdg's per-folder label files into one table and zips the lot
for Colab.

## Fine-tuning

The notebook follows keras-ocr's fine-tuning example with a few changes:

- **Alphabet.** Digits, lowercase letters and punctuation. Because that differs
  from the pretrained alphabet, keras-ocr keeps only the CRNN backbone weights
  and trains a fresh output layer.
- **Split.** Labels lowercased, then 70/30 train/test stratified by category,
  so every category shows up in both.
- **Augmentation.** Gamma contrast between 0.25 and 3.0.
- **Stopping.** Early stopping on validation loss with patience 10, keeping
  the best checkpoint.

```python
recognizer = keras_ocr.recognition.Recognizer(alphabet=recognizer_alphabet)
recognizer.compile()
# "Provided alphabet does not match pretrained alphabet. Using backbone weights only."
```

## Results

On the font-size-30 set, evaluated on 7,200 held-out images:

| test set | exact match |
|---|---|
| all six categories | 83.7% |

A string only counts if every character is right, so reading `a8ti0wb` as
`a8tiowb` is a miss. The notebook notes two confusions that survived training:
`l` read as `i`, and `o` read as `0`.

Two limits on that number. It's synthetic data scored against synthetic data,
and the model never saw a real scan. And the notebook also reports 87.5% on the
words-with-symbols category alone, but that figure comes from a fresh
re-split, so its test images may overlap training. I wouldn't quote it.

## What I'd change

Hold out a small set of real images from day one, even fifty, and report on
those. Synthetic accuracy tells you the training loop works; it doesn't tell
you the model reads your documents.
