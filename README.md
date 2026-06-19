# Pashto TTS — Text-to-Speech for Pashto

A Pashto Text-to-Speech system trained from scratch using the **MB-iSTFT-VITS2** neural architecture on the **Mangal Paktika accent**.

**98% word accuracy** — verified by native Pashto speakers.

> **Just want to hear it?** Pre-generated audio samples are in the [`samples/`](samples/) folder. Download and listen — no setup needed.

---

## Table of Contents

- [Listen to Samples](#listen-to-samples)
- [What This Is](#what-this-is)
- [Setup — 5 Steps](#setup--5-steps)
- [Run the Gradio Demo](#run-the-gradio-demo)
- [Use in Python (no UI)](#use-in-python-no-ui)
- [How It Was Built](#how-it-was-built)
- [Supported Characters](#supported-pashto-characters)
- [Retrain From Scratch](#retrain-from-scratch)
- [File Guide](#file-guide)
- [Limitations](#limitations)

---

## Listen to Samples

All samples are in [`samples/`](samples/). Generated using checkpoint `G_314000`, `noise_scale=0.4`.

| File | Pashto Text | English Meaning |
|------|-------------|----------------|
| `01_salam.wav` | سلام | Hello |
| `02_manana.wav` | مننه | Thank you |
| `03_greeting.wav` | سلام، زه ستاسو سره مرسته کولی شم | Hello, I can help you |
| `04_school.wav` | زه هره ورځ مکتب ته ځم | I go to school every day |
| `05_pashto_history.wav` | دا زموږ د پښتو ژبې تاریخ دی | This is the history of our Pashto language |
| `06_afghanistan.wav` | افغانستان یو ښکلی هېواد دی | Afghanistan is a beautiful country |
| `07_peace.wav` | سوله او امنیت ډېر مهم دي | Peace and security are very important |
| `08_weather.wav` | نن ورځ هوا ډېره ښه ده | Today the weather is very good |
| `09_pashtun.wav` | زه پښتون یم او پښتو خبرې کوم | I am Pashtun and I speak Pashto |
| `10_education.wav` | زه غواړم چې ټول ماشومان مکتب ته ولاړ شي | I want all children to go to school |
| `11_long_history.wav` | زموږ د پښتو ژبې تاریخ ډېر پخوانی دی... | The history of our Pashto language is very old... |
| `12_country.wav` | افغانستان یو هېواد دی چې... | Afghanistan is a country known for... |

---

## What This Is

### The Problem
There are very few high-quality Text-to-Speech tools for Pashto. Existing tools don't handle the Pashto script accurately, and no publicly available model covers the **Mangal Paktika accent** specifically.

### What We Did
1. Collected **5,457 clean audio recordings** from a native Mangal Paktika speaker — 8.4 hours total
2. Trained the **MB-iSTFT-VITS2** model end-to-end on this data (text goes in, speech comes out)
3. Ran **1,000 training epochs** (~36 hours on an RTX 4060 Ti GPU)
4. Evaluated multiple checkpoints and selected the best one based on audio quality

### The Result
Type any Pashto text → get natural-sounding Mangal Paktika speech within seconds.

---

## Setup — 5 Steps

### What you need
- Python 3.10 or newer
- An NVIDIA GPU with at least 4 GB VRAM
- CUDA 11.8 or newer ([install guide](https://developer.nvidia.com/cuda-downloads))

---

### Step 1 — Clone this repo

```bash
git clone https://github.com/HussainM10/pashto-tts.git
cd pashto-tts
```

---

### Step 2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs PyTorch, Gradio, librosa, and a few other packages the model needs.

> **Note:** If you have a specific CUDA version, install PyTorch manually first from [pytorch.org](https://pytorch.org/get-started/locally/) before running the above command.

---

### Step 3 — Download the trained model

The model file (`G_314000.pth`, 458 MB) is too large for GitHub, so it is hosted in [GitHub Releases](https://github.com/HussainM10/pashto-tts/releases).

**Option A — run the download script (Linux / WSL / Mac):**

```bash
bash download_model.sh
```

**Option B — download manually:**

1. Go to the [Releases page](https://github.com/HussainM10/pashto-tts/releases/latest)
2. Download `G_314000.pth`
3. Move the file into the `model/` folder

After this, your folder should look like:

```
model/
└── G_314000.pth   ← must be here
```

---

### Step 4 — Run the demo

```bash
python app.py
```

Then open **http://localhost:7860** in your browser.

---

### Step 5 — Type Pashto and click Generate

Paste any Pashto text into the box and click **Generate**. Audio plays directly in the browser.

---

## Run the Gradio Demo

```bash
python app.py
# Opens at http://localhost:7860
```

### Controls

| Control | Default | What it does |
|---------|---------|--------------|
| `noise_scale` | **0.4** | Controls expressiveness / variation in speech. Lower = more robotic. Higher = more varied. |
| `noise_scale_w` | 0.7 | Controls duration variation between words |
| `length_scale` | 1.0 | Speaking speed. Try `0.8` for faster, `1.2` for slower |

**Recommended starting point:** Keep all defaults. Only adjust `noise_scale` if the voice sounds too flat or too chaotic.

---

## Use in Python (no UI)

If you want to generate audio from a script instead of the web interface:

```python
import torch
import json
import numpy as np
import scipy.io.wavfile as wavfile
from text import text_to_sequence
from text.symbols import symbols
from models import SynthesizerTrn
import commons

# --- Load config ---
with open("configs/pashto.json") as f:
    hps = json.load(f)

# --- Load model ---
net_g = SynthesizerTrn(
    len(symbols),
    80,
    hps["train"]["segment_size"] // hps["data"]["hop_length"],
    n_speakers=hps["data"]["n_speakers"],
    **hps["model"],
).cuda()

checkpoint = torch.load("model/G_314000.pth", map_location="cpu")
net_g.load_state_dict(checkpoint["model"])
net_g.eval()

# --- Generate speech ---
text = "سلام، زه ستاسو سره مرسته کولی شم"

seq = text_to_sequence(text, ["pashto_cleaners"])
seq = commons.intersperse(seq, 0)  # add blank tokens between characters

x = torch.LongTensor(seq).unsqueeze(0).cuda()
x_len = torch.LongTensor([len(seq)]).cuda()
sid = torch.LongTensor([0]).cuda()

with torch.no_grad():
    audio = net_g.infer(
        x, x_len, sid=sid,
        noise_scale=0.4,
        noise_scale_w=0.7,
        length_scale=1.0,
    )[0][0, 0]

# --- Save to file ---
audio_np = audio.cpu().numpy()
wavfile.write("output.wav", 22050, audio_np)
print("Saved: output.wav")
```

---

## How It Was Built

### Training Data

| Property | Value |
|----------|-------|
| Source | Mangal Paktika accent (native speaker recordings) |
| Number of clips | 5,457 WAV files |
| Total duration | 8.4 hours |
| Average clip length | ~5.5 seconds |
| Sample rate | 22,050 Hz, mono |
| Speaker | Single speaker |

### Model: MB-iSTFT-VITS2

**VITS2** stands for *Variational Inference with adversarial learning for end-to-end Text-to-Speech v2*. It is an end-to-end neural model — meaning it takes raw text and outputs raw audio with no manual intermediate steps.

**MB-iSTFT** (Multi-Band inverse Short-Time Fourier Transform) is an audio decoder that produces higher-quality output at faster speed than standard vocoders.

| Property | Value |
|----------|-------|
| Architecture | MB-iSTFT-VITS2 |
| Parameters | ~40 million |
| Input | Pashto Unicode text |
| Output | 22,050 Hz mono WAV |

### Training Details

| Setting | Value |
|---------|-------|
| Total epochs | 1,000 |
| Best checkpoint | Step 314,000 (epoch ~969) |
| Batch size | 16 |
| Learning rate | 2e-4 with decay |
| GPU | NVIDIA RTX 4060 Ti 16 GB |
| Total training time | ~36 hours |
| Framework | PyTorch |

Training was done using the standard VITS2 GAN setup: a Generator (the TTS model) is trained against a Discriminator (a model that tries to tell real speech from generated). Only the Generator is needed for inference.

### Why VITS2?

Three architectures were tested before arriving at VITS2:

| Approach | Result |
|----------|--------|
| **VITS2 (this repo)** | 98% accuracy, fast inference (~1s per sentence) |
| Orpheus TTS (LoRA on Llama-3B) | Slow (~107s per sentence), not fully trained |
| F5-TTS | Only trained to 28% of planned epochs |

VITS2 was selected for its combination of quality, speed, and stability.

---

## Supported Pashto Characters

The model supports **59 Pashto characters** from the Mangal Paktika subset of the Pashto script. See [`text/symbols.py`](text/symbols.py) for the exact list.

This includes standard Pashto characters as well as Pashto-specific letters:
`ښ ځ څ ړ ږ ګ ڼ ۍ ې`

**Characters outside this set are silently ignored.** If a word is not being spoken, it likely contains unsupported characters.

Input text should be standard Pashto Unicode. No romanization or phoneme conversion is needed.

---

## Retrain From Scratch

Follow these steps if you want to train your own version (different accent, more data, etc.).

### 1. Prepare your audio data

You need WAV files (mono, 22050 Hz) and a text file pairing each file with its transcript:

```
# filelists/your_train.txt  (pipe-separated)
path/to/audio01.wav|پښتو متن
path/to/audio02.wav|بل متن دلته
```

Recommended split: 90% train, 5% validation, 5% test.

### 2. Update the config

Open `configs/pashto.json` and change:
- `"training_files"` → path to your train file
- `"validation_files"` → path to your val file

### 3. Update the character set (if needed)

If your data contains characters not in the current 59-character set, add them to `text/symbols.py`. Make sure `text/cleaners.py` handles them correctly.

### 4. Start training

```bash
python scripts/train_ms.py -c configs/pashto.json -m my_pashto_model
```

Checkpoints are saved to `logs/my_pashto_model/` every 2,000 steps.

### 5. Monitor training (TensorBoard)

```bash
tensorboard --logdir logs/my_pashto_model
```

Watch the **mel loss** — it should steadily decrease toward ~15–20 for decent quality.

### 6. Run inference on a checkpoint

```bash
python scripts/inference.py \
  --checkpoint logs/my_pashto_model/G_XXXXXX.pth \
  --config configs/pashto.json \
  --text "سلام" \
  --output output.wav
```

### Training tips

- 1,000 epochs is a good target for a ~8-hour dataset
- The best checkpoint is often NOT the last one — evaluate several checkpoints around epoch 900–970
- Keep `noise_scale=0.4` for evaluation — it tends to produce the clearest output

---

## File Guide

```
pashto-tts/
│
├── app.py                   ← Gradio web demo — run this
├── download_model.sh        ← Downloads the model from GitHub Releases
├── requirements.txt         ← Python packages needed
│
├── model/
│   └── G_314000.pth         ← Trained model (458 MB, download separately)
│
├── configs/
│   └── pashto.json          ← Model architecture + training settings
│
├── text/                    ← Pashto text processing
│   ├── symbols.py           ← 59-character Pashto vocabulary
│   ├── cleaners.py          ← Text normalization (pashto_cleaners)
│   └── __init__.py          ← text_to_sequence() — converts text to token IDs
│
├── samples/                 ← Pre-generated audio files (no setup needed)
│   ├── samples.txt          ← What each file says
│   └── 01_salam.wav ... 12_country.wav
│
├── filelists/               ← Train/val/test splits (needed for retraining)
│   ├── mangal_train.txt
│   ├── mangal_val.txt
│   └── mangal_test.txt
│
├── scripts/                 ← Extra scripts for training and evaluation
│   ├── train_ms.py          ← Training script
│   ├── inference.py         ← CLI inference (generate single audio file)
│   ├── evaluate_all.py      ← Run evaluation across test sentences
│   └── preprocess.py        ← Data preprocessing (resample, filter)
│
└── [core model code]        ← Required by app.py — do not modify unless retraining
    ├── models.py            ← SynthesizerTrn — the main model class
    ├── modules.py           ← Building blocks (residual blocks, etc.)
    ├── attentions.py        ← Attention layers (Transformer blocks)
    ├── mel_processing.py    ← Converts audio to/from mel spectrogram
    ├── transforms.py        ← Normalizing flows
    ├── stft.py              ← Short-time Fourier transform
    ├── stft_loss.py         ← STFT loss (used during training)
    ├── pqmf.py              ← Multi-band audio filter bank
    ├── losses.py            ← Training loss functions
    ├── data_utils.py        ← Data loading and batching
    ├── commons.py           ← Shared utilities
    ├── utils.py             ← Checkpoint loading, logging
    ├── S_monotonic_align.py ← Duration alignment
    └── monotonic_align/     ← Cython-compiled alignment module
```

---

## Limitations

| Limitation | Detail |
|------------|--------|
| **Accent** | Speaks with the Mangal Paktika accent only. May sound unnatural for speakers of Kandahari, Peshawar, or other Pashto dialects. |
| **Vocabulary** | Works best with words seen during training. Unusual or foreign words may sound distorted. |
| **Consonants** | Occasional confusion between similar consonants (پ/ب, ټ/ت, ښ/خ) in long or unseen sentences. |
| **Character set** | Only 59 supported characters. Anything else is silently dropped. |
| **Speaker** | Single-speaker model. No voice cloning or speaker control. |
| **GPU** | Inference requires a CUDA GPU. CPU inference works but is very slow. |
| **Script direction** | Input must be standard right-to-left Pashto Unicode. Latin or mixed text is not supported. |

---

## License

See [LICENSE](LICENSE) for the base VITS2 code license.

The trained model weights and audio samples are released for research and educational use.

---

*Trained at [BarakatPay](https://barakatpay.com) · Architecture: [MB-iSTFT-VITS2](https://github.com/MasayaKawamura/MB-iSTFT-VITS2)*
