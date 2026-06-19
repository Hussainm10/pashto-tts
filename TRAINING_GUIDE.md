# Training Guide — How to Train a Pashto TTS Model from Scratch

This guide explains everything step by step, assuming you are new to training TTS models.
If you already have experience, skip to [Step 3](#step-3--prepare-your-file-lists).

---

## How TTS Works (Quick Explanation)

Traditional TTS has two separate stages:
1. **Acoustic model** — converts text into a spectrogram (a visual representation of sound)
2. **Vocoder** — converts the spectrogram into actual audio

**VITS2 skips this split.** It is an end-to-end model: text goes in, audio comes out directly.
Internally it still does both steps, but they are trained together, which produces better quality.

Here is what VITS2 does:
```
Input text
    ↓
Convert each character to an ID number (using the vocabulary in text/symbols.py)
    ↓
Transformer encoder: understand the relationships between characters
    ↓
Duration predictor: decide how long each character sounds
    ↓
Flow-based decoder: convert acoustic features to audio
    ↓
Multi-Band iSTFT vocoder: produce the final waveform
    ↓
Output WAV audio
```

The model is trained with two competing networks:
- **Generator (G)** — the actual TTS model, learns to produce realistic speech
- **Discriminator (D)** — a judge that tries to tell real speech from fake. Forces G to improve.

This adversarial setup is what makes the output sound natural rather than robotic.
The Discriminator is only needed during training. For inference, only `G_XXXXXX.pth` is used.

---

## What You Need Before Training

### Hardware
- NVIDIA GPU with at least 8 GB VRAM (16 GB recommended)
- 50+ GB disk space for checkpoints

### Data requirements
| Requirement | Minimum | Recommended |
|------------|---------|-------------|
| Audio clips | 1,000 | 5,000+ |
| Total hours | 1 hour | 5–10 hours |
| Speaker(s) | Single speaker | Single speaker (multi-speaker is harder) |
| Audio quality | Clean, no background noise | Studio quality |
| Sample rate | Any (we will resample) | 22,050 Hz |

> **Key rule:** All audio must come from the **same speaker** in a **consistent environment**.
> Mixed speakers or recordings in different rooms will produce garbled output.

### Software
```bash
pip install -r requirements.txt
pip install librosa soundfile tqdm
```

---

## Step 1 — Collect Audio Data

You need:
1. **Audio files** — WAV format (MP3 works too, we will convert)
2. **Transcripts** — the exact text spoken in each file

Good free sources for Pashto:
- [Mozilla Common Voice](https://commonvoice.mozilla.org/ps) — pashto dataset, free download
- Your own recordings — use Audacity or any microphone

**Quality checklist for each recording:**
- [ ] One speaker only
- [ ] No background music or crowd noise
- [ ] No clipping (audio should not hit 0 dB)
- [ ] Each file is 1–15 seconds (very short or very long clips cause problems)
- [ ] Transcript exactly matches what is spoken (no extra words, no missing words)

---

## Step 2 — Preprocess Audio

All audio must be:
- **Mono** (not stereo)
- **22,050 Hz** sample rate
- **WAV format**

Use this script to convert a folder of files:

```python
# resample.py — run this on your raw audio folder
import os
import librosa
import soundfile as sf

INPUT_DIR  = "raw_audio/"     # your original files
OUTPUT_DIR = "wavs_22k/"      # where processed files go
TARGET_SR  = 22050

os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith((".wav", ".mp3", ".flac")):
        continue
    src = os.path.join(INPUT_DIR, fname)
    dst = os.path.join(OUTPUT_DIR, os.path.splitext(fname)[0] + ".wav")
    audio, sr = librosa.load(src, sr=TARGET_SR, mono=True)
    sf.write(dst, audio, TARGET_SR)
    print(f"Converted: {fname}")

print("Done.")
```

```bash
python resample.py
```

---

## Step 3 — Prepare Your File Lists

VITS2 reads data from plain text files. Each line has three fields separated by `|`:

```
path/to/audio.wav|0|Pashto text here
```

- **Field 1**: full path to the WAV file
- **Field 2**: speaker ID — always `0` for single-speaker training
- **Field 3**: the Pashto transcript

**Example from this project:**
```
/mnt/e/data/wavs_22k/clip001.wav|0|سلام، زه ستاسو سره مرسته کولی شم
/mnt/e/data/wavs_22k/clip002.wav|0|افغانستان یو ښکلی هېواد دی
/mnt/e/data/wavs_22k/clip003.wav|0|زه هره ورځ مکتب ته ځم
```

**Split your data into three files:**
- `filelists/train.txt` — 90% of your clips (used for training)
- `filelists/val.txt` — 5% (used to check if the model is improving)
- `filelists/test.txt` — 5% (used for final evaluation only)

Use absolute paths. If your audio is at `/home/user/data/wavs_22k/`, the paths should start with that.

**Quick Python script to build the lists from a CSV:**

```python
# build_filelists.py
import csv, random, os

CSV_FILE   = "transcripts.csv"   # columns: filename, text
AUDIO_DIR  = "/absolute/path/to/wavs_22k/"
OUTPUT_DIR = "filelists/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(CSV_FILE, encoding="utf-8") as f:
    rows = [(r["filename"], r["text"]) for r in csv.DictReader(f)]

random.shuffle(rows)
n = len(rows)
train = rows[:int(n * 0.90)]
val   = rows[int(n * 0.90):int(n * 0.95)]
test  = rows[int(n * 0.95):]

for split, data in [("train", train), ("val", val), ("test", test)]:
    with open(f"{OUTPUT_DIR}/{split}.txt", "w", encoding="utf-8") as f:
        for fname, text in data:
            wav = os.path.join(AUDIO_DIR, fname.replace(".mp3", ".wav"))
            f.write(f"{wav}|0|{text}\n")

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
```

---

## Step 4 — Update the Config

Open `configs/pashto.json` and change these two lines:

```json
"training_files": "filelists/train.txt",
"validation_files": "filelists/val.txt",
```

Replace with the paths to your files. Everything else can stay the same for a first run.

**Optional adjustments:**

| Setting | Default | When to change |
|---------|---------|---------------|
| `batch_size` | 16 | Lower to 8 if you get out-of-memory errors |
| `epochs` | 1000 | Increase for more data, decrease to test quickly |
| `segment_size` | 8192 | Lower to 4096 if out-of-memory |
| `sampling_rate` | 22050 | Only change if your audio is a different rate |

---

## Step 5 — Update the Character Set

If you are training on **Pashto with the same Mangal Paktika accent**, skip this step — the existing `text/symbols.py` covers 59 Pashto characters.

If you are training on **a different language or dialect**:

1. Open `text/symbols.py` and find the `_pashto_characters` list
2. Replace it with the characters in your language
3. Make sure every character that appears in your transcripts is in this list

How to find all unique characters in your data:
```python
# find_chars.py
chars = set()
for line in open("filelists/train.txt", encoding="utf-8"):
    text = line.strip().split("|")[2]
    chars.update(text)
print(sorted(chars))
```

Any character not in `symbols.py` will be silently ignored during training and inference.

---

## Step 6 — Train

Run from the **repo root** (not from inside `scripts/`):

```bash
python scripts/train_ms.py -c configs/pashto.json -m my_pashto_model
```

- `-c` — path to config file
- `-m` — name for your run (checkpoints save to `logs/my_pashto_model/`)

Checkpoints are saved every 2,000 steps. On an RTX 4060 Ti (16GB), 1,000 epochs on 5,457 clips took about **36 hours**.

### What you will see during training

```
Epoch 1/1000  Step 200/3629  | mel_loss=85.3  kl_loss=1.2  dur_loss=4.1
Epoch 1/1000  Step 400/3629  | mel_loss=72.1  kl_loss=1.1  dur_loss=3.8
...
Epoch 50/1000  Step 181450   | mel_loss=28.4  kl_loss=0.9  dur_loss=1.2
...
Epoch 500/1000 Step 1814500  | mel_loss=16.1  kl_loss=0.8  dur_loss=0.6
```

**Mel loss** is the most important number — it measures how close the generated spectrogram is to the real one.

| Mel loss | What it means |
|----------|--------------|
| 80–100 | Training just started, output is noise |
| 40–60 | Rough speech shape forming |
| 20–30 | Recognizable speech, some errors |
| 15–20 | Good quality — evaluate checkpoints here |
| Below 15 | Can overfit — test carefully |

---

## Step 7 — Monitor Training (TensorBoard)

In a separate terminal while training runs:

```bash
tensorboard --logdir logs/my_pashto_model
# Open http://localhost:6006 in browser
```

Watch these graphs:
- **loss/mel** — should steadily fall
- **loss/kl** — small number, stays roughly flat
- **loss/dur** — duration loss, should also fall

If mel loss stops falling for many epochs, training has plateaued. You can stop early.

---

## Step 8 — Pick the Best Checkpoint

The last checkpoint is often NOT the best. Models sometimes start to overfit after epoch ~900.

Test several checkpoints around epochs 800–1000:

```bash
# Test checkpoint at step 280000
python scripts/inference.py \
  --checkpoint logs/my_pashto_model/G_280000.pth \
  --config configs/pashto.json \
  --text "سلام، زه ستاسو سره مرسته کولی شم" \
  --output test_280k.wav

# Test checkpoint at step 300000
python scripts/inference.py \
  --checkpoint logs/my_pashto_model/G_300000.pth \
  --config configs/pashto.json \
  --text "سلام، زه ستاسو سره مرسته کولی شم" \
  --output test_300k.wav
```

Listen and compare. The best checkpoint for this project was **G_314000** (step 314,000, epoch ~969).

---

## Common Errors and Fixes

These are the exact errors we hit when training this model and how we fixed them.

---

### Error: `librosa.filters.mel() got unexpected keyword argument`

**Cause:** Older librosa API.

**Fix** (already applied in this repo in `mel_processing.py`):
```python
# Old (broken):
mel_basis = librosa.filters.mel(hps.data.sampling_rate, ...)

# Fixed:
mel_basis = librosa.filters.mel(sr=hps.data.sampling_rate, n_fft=...)
```

---

### Error: `RuntimeError: stft() received a complex tensor`

**Cause:** PyTorch 2.x changed the default for `return_complex`.

**Fix** (already applied in `stft_loss.py`):
```python
# Old (broken):
x_stft = torch.stft(x, fft_size, hop_size, win_length, window)

# Fixed:
x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=True)
```

---

### Error: `AttributeError: tostring_rgb`

**Cause:** Newer matplotlib removed `tostring_rgb`.

**Fix** (already applied in `utils.py`):
```python
# Old (broken):
data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)

# Fixed:
data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
```

---

### Error: `RuntimeError: Function ... returned nan values` during training

**Cause:** PyTorch anomaly detection slowing things down and sometimes causing NaN spirals.

**Fix** (already applied in `scripts/train_ms.py`):
```python
torch.autograd.set_detect_anomaly(False)
```

---

### Error: Out of memory (OOM) during training

**Fix options (try in order):**
1. Reduce `batch_size` in config from 16 to 8
2. Reduce `segment_size` from 8192 to 4096
3. Enable `fp16_run: true` in config (half precision)

---

### Error: Audio output sounds like noise even after many epochs

**Possible causes:**
- Audio files have background noise — re-filter the dataset
- Transcripts don't match the audio — check alignment manually
- Characters in transcripts not in `symbols.py` — run the `find_chars.py` script above
- Learning rate too high — try halving it in config

---

### Error: `ImportError: cannot import name 'maximum_path_c' from 'monotonic_align.core'`

**Cause:** The Cython extension isn't compiled. This repo uses a pure-Python fallback (`S_monotonic_align.py`) so this should not occur. If it does:

```bash
cd monotonic_align
python setup.py build_ext --inplace
cd ..
```

---

## Training Tips

1. **Start with a small test run first.** Set `epochs: 10` in config, make sure training starts without errors, then set it back to 1000.

2. **More data is almost always better.** Going from 1 hour to 8 hours of clean audio will make a bigger difference than any hyperparameter change.

3. **Audio quality matters more than quantity.** 2 hours of clean studio audio beats 8 hours of noisy phone recordings.

4. **Evaluate early.** After epoch 100–200, generate a few samples to check the model is learning. If it sounds like random noise after 200 epochs, something is wrong.

5. **Save checkpoints frequently.** The default saves every 2,000 steps. Training can crash; you don't want to lose everything.

6. **The best checkpoint is usually not the last one.** Evaluate checkpoints from the last 10–20% of training and pick by ear.

7. **Use `noise_scale=0.4` for evaluation.** Higher values produce more variation but can introduce distortion, making it hard to compare checkpoints fairly.

---

## How This Model Was Trained (Reference)

| Step | Detail |
|------|--------|
| Data source | Mozilla Common Voice Pashto (Mangal Paktika accent) |
| Raw clips | 10,066 recordings → filtered to 5,457 clean clips |
| Filtering criteria | Duration 1–15s, transcript length 5–100 chars, no silence |
| Preprocessing | Resampled to 22,050 Hz mono WAV |
| Training split | 95% train, 5% validation |
| Total training time | ~36 hours on RTX 4060 Ti 16GB |
| Best checkpoint | G_314000 (epoch ~969 out of 1000) |
| Evaluation | Native speaker scoring + Whisper ASR word error rate |
| Final score | 98% word accuracy |
