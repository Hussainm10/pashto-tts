#!/usr/bin/env python3
"""
Pashto TTS — command-line inference
Run from the repo root:

    python scripts/inference.py --text "سلام" --output output.wav
    python scripts/inference.py --text "افغانستان یو ښکلی هېواد دی" --output afghanistan.wav --noise_scale 0.4
"""
import argparse
import json
import os
import sys

import torch
import scipy.io.wavfile as wavfile

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text import text_to_sequence
from text.symbols import symbols
from models import SynthesizerTrn
import commons


def main():
    parser = argparse.ArgumentParser(description="Generate Pashto speech from text")
    parser.add_argument("--text",          required=True,  help="Pashto text to speak")
    parser.add_argument("--output",        default="output.wav", help="Output WAV file path")
    parser.add_argument("--checkpoint",    default="model/G_314000.pth", help="Path to model checkpoint")
    parser.add_argument("--config",        default="configs/pashto.json", help="Path to config file")
    parser.add_argument("--noise_scale",   type=float, default=0.4,  help="Voice expressiveness (default 0.4)")
    parser.add_argument("--noise_scale_w", type=float, default=0.7,  help="Duration variation (default 0.7)")
    parser.add_argument("--length_scale",  type=float, default=1.0,  help="Speed: <1 faster, >1 slower")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        hps = json.load(f)

    # Load model
    print(f"Loading model from {args.checkpoint} ...")
    net_g = SynthesizerTrn(
        len(symbols),
        80,
        hps["train"]["segment_size"] // hps["data"]["hop_length"],
        n_speakers=hps["data"]["n_speakers"],
        **hps["model"],
    ).cuda().eval()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    net_g.load_state_dict(checkpoint["model"])
    print("Model loaded.")

    # Convert text to token IDs
    seq = text_to_sequence(args.text, hps["data"]["text_cleaners"])
    seq = commons.intersperse(seq, 0)  # blank token between each character

    x = torch.LongTensor(seq).unsqueeze(0).cuda()
    x_len = torch.LongTensor([len(seq)]).cuda()
    sid = torch.LongTensor([0]).cuda()

    # Generate audio
    with torch.no_grad():
        audio = net_g.infer(
            x, x_len, sid=sid,
            noise_scale=args.noise_scale,
            noise_scale_w=args.noise_scale_w,
            length_scale=args.length_scale,
        )[0][0, 0].cpu().numpy()

    # Save
    wavfile.write(args.output, hps["data"]["sampling_rate"], audio)
    print(f"Saved: {args.output}  ({len(audio) / hps['data']['sampling_rate']:.2f} seconds)")


if __name__ == "__main__":
    main()
