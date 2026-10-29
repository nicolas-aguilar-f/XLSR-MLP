"""Classify audio files as bona fide (real) or spoof (synthetic) Spanish speech.

Examples:
    python predict.py audio.wav
    python predict.py real.wav fake.flac --device cuda
    python predict.py audio.mp3 --json
"""
import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
from transformers.utils import logging as hf_logging

from xlsr_mlp import SpoofDetector

DEFAULT_CHECKPOINT = Path(__file__).parent / "models" / "mlp_antispoofing_spanish.pth"
VERDICTS = {"bonafide": "BONA FIDE (real)", "spoof": "SPOOF (synthetic)"}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog=__doc__.split("\n\n", 1)[1],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", nargs="+", type=Path,
                        help="audio file(s) to classify (WAV, FLAC, MP3, OGG, ...)")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="trained MLP checkpoint "
                             "(default: models/mlp_antispoofing_spanish.pth)")
    parser.add_argument("--device", default="auto",
                        help='"auto", "cpu", "cuda" or "cuda:N" (default: auto)')
    parser.add_argument("--seed", type=int, default=1234,
                        help="seed of the neutralization noise (default: %(default)s)")
    parser.add_argument("--json", action="store_true",
                        help="print one JSON object per file instead of a table")
    args = parser.parse_args()

    missing = [str(p) for p in args.audio if not p.is_file()]
    if missing:
        parser.error("file not found: " + ", ".join(missing))

    # Hide the "unused weights" warnings of loading XLS-R without its
    # pre-training heads (they are expected) and the loading progress bars.
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    print("Loading XLS-R 300M (about 1.2 GB is downloaded on the first run)...",
          file=sys.stderr)
    detector = SpoofDetector(args.checkpoint, args.device)

    if not args.json:
        print(f"\n{'Prediction':<20} {'Spoof score':>11}   File")
    failed = False
    for path in args.audio:
        try:
            result = detector.predict(path, seed=args.seed)
        except (sf.SoundFileError, ValueError) as err:
            print(f"error: cannot process {path}: {err}", file=sys.stderr)
            failed = True
            continue
        if args.json:
            print(json.dumps(result))
        else:
            print(f"{VERDICTS[result['label']]:<20} "
                  f"{result['spoof_score']:>11.4f}   {path}")

    if not args.json:
        print(f"\nFiles with a spoof score >= {detector.threshold:.4f} "
              "(EER threshold of the validation set) are flagged as spoof.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
