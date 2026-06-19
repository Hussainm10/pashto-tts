"""
Post-Training Evaluation Script
================================
Tests all available checkpoints, finds the best one, runs Whisper on all sentences.
Run this after training completes.

Usage: python3 evaluate_all.py
"""

import torch
import json
import numpy as np
import os
import glob
import warnings
from scipy.io import wavfile
from text import text_to_sequence
from text.symbols import symbols
from models import SynthesizerTrn
import commons

warnings.filterwarnings("ignore")

CONFIG = "configs/pashto.json"
LOG_DIR = "logs/mangal"
OUTPUT_DIR = "test_outputs/final_eval"

TEST_SENTENCES = [
    ("greeting", "سلام، زه ستاسو سره مرسته کولی شم"),
    ("afghanistan", "افغانستان یو ښکلی هېواد دی"),
    ("pashto", "دا زموږ د پښتو ژبې تاریخ دی"),
    ("school", "زه هره ورځ مکتب ته ځم"),
    ("peace", "سوله او امنیت ډېر مهم دي"),
    ("longer", "زموږ د پښتو ژبې تاریخ ډېر پخوانی دی او دا ژبه د نړۍ په مختلفو برخو کې ویل کیږي"),
    ("welcome", "په خیر راغلاست"),
    ("question", "تاسو څنګه یاست"),
    ("thanks", "مننه ستاسو د مرستې لپاره"),
    ("country", "افغانستان یو هېواد دی چې د لوړو غرونو او شنو درو لپاره مشهور دی"),
]

NOISE_SCALES = [0.4, 0.5, 0.667]


def load_model(ckpt_path, hps):
    net_g = SynthesizerTrn(
        len(symbols), 80,
        hps["train"]["segment_size"] // hps["data"]["hop_length"],
        n_speakers=hps["data"]["n_speakers"],
        **hps["model"],
    ).cuda()
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    net_g.load_state_dict(checkpoint["model"])
    net_g.eval()
    return net_g


def generate(net_g, text, hps, noise_scale=0.5):
    seq = text_to_sequence(text, ["pashto_cleaners"])
    if hps["data"]["add_blank"]:
        seq = commons.intersperse(seq, 0)

    x = torch.LongTensor(seq).unsqueeze(0).cuda()
    x_len = torch.LongTensor([len(seq)]).cuda()
    sid = torch.LongTensor([0]).cuda()

    with torch.no_grad():
        audio = net_g.infer(x, x_len, sid=sid, noise_scale=noise_scale, noise_scale_w=0.8, length_scale=1.0)

    audio_np = audio[0][0].data.cpu().float().numpy().squeeze()
    return np.clip(audio_np, -1.0, 1.0)


def whisper_eval(audio_np, sr=22050):
    import whisper
    import librosa

    audio_16k = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)
    model = whisper_eval._model
    result = model.transcribe(audio_16k, language="ps")
    return result["text"].strip()


def main():
    with open(CONFIG) as f:
        hps = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all checkpoints
    ckpt_files = sorted(glob.glob(os.path.join(LOG_DIR, "G_*.pth")))
    # Also check backup
    backup_ckpts = sorted(glob.glob(os.path.join(LOG_DIR, "backup_ckpts", "G_*.pth")))
    ckpt_files = backup_ckpts + ckpt_files

    print(f"Found {len(ckpt_files)} checkpoints")
    for f in ckpt_files:
        step = int(os.path.basename(f).replace("G_", "").replace(".pth", ""))
        print(f"  G_{step} (epoch ~{step // 324})")

    # Load Whisper
    import whisper
    print("\nLoading Whisper...")
    whisper_eval._model = whisper.load_model("base")

    # Test each checkpoint with greeting sentence first (quick screen)
    print("\n" + "=" * 70)
    print("  PHASE 1: Quick screen — greeting sentence, all checkpoints")
    print("=" * 70)

    best_ckpt = None
    best_score = ""
    results = {}

    for ckpt_path in ckpt_files:
        step = int(os.path.basename(ckpt_path).replace("G_", "").replace(".pth", ""))
        print(f"\n  G_{step}:")

        net_g = load_model(ckpt_path, hps)

        for ns in NOISE_SCALES:
            audio = generate(net_g, TEST_SENTENCES[0][1], hps, noise_scale=ns)
            wtext = whisper_eval(audio)
            tag = f"G{step}_ns{ns}"
            results[tag] = {"whisper": wtext, "step": step, "ns": ns}
            print(f"    ns={ns}: {wtext[:60]}")

        del net_g
        torch.cuda.empty_cache()

    # Find best checkpoint/noise combo based on greeting
    # (manual review needed — just print all for comparison)

    # Phase 2: Full eval on top 3 checkpoints
    print("\n" + "=" * 70)
    print("  PHASE 2: Full eval — all sentences, best 3 checkpoints, ns=0.5")
    print("=" * 70)

    # Use last 3 checkpoints + any backup
    eval_ckpts = ckpt_files[-3:]

    for ckpt_path in eval_ckpts:
        step = int(os.path.basename(ckpt_path).replace("G_", "").replace(".pth", ""))
        print(f"\n  --- G_{step} (epoch ~{step // 324}) ---")

        net_g = load_model(ckpt_path, hps)

        for name, text in TEST_SENTENCES:
            audio = generate(net_g, text, hps, noise_scale=0.5)
            audio_int16 = (audio * 32767).astype(np.int16)

            fpath = os.path.join(OUTPUT_DIR, f"G{step}_{name}.wav")
            wavfile.write(fpath, 22050, audio_int16)

            wtext = whisper_eval(audio)
            print(f"  [{name:>12}] {wtext[:70]}")
            print(f"  {'':>14} expected: {text[:70]}")
            print()

        del net_g
        torch.cuda.empty_cache()

    print("=" * 70)
    print(f"  Audio files saved to: {OUTPUT_DIR}/")
    print(f"  Listen to them and pick the best checkpoint.")
    print("=" * 70)


if __name__ == "__main__":
    main()
