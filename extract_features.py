"""Extract XLS-R embeddings for every split of the HISPASpoof detection subset.

Each audio file is loaded (mono, 16 kHz, first 15 s), neutralized with white
noise at 36 dB SNR and mapped to a 2048-D vector ([mean, std] of the last
XLS-R layer). Each split is saved as <output-dir>/<split>.npz with the arrays
X (N, 2048), y (N,) and filenames (N,).

Expected dataset layout (the "detection" folder of HISPASpoof):

    <data-dir>/
    |-- protocols/
    |   |-- train_metadata.csv      columns: filename, label (0 real, 1 spoof)
    |   |-- val_metadata.csv
    |   `-- test_metadata.csv
    |-- train/
    |-- val/
    `-- test/

Example:
    python extract_features.py --data-dir /data/HISPASpoof/detection --output-dir features
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from transformers.utils import logging as hf_logging

from xlsr_mlp import XLSREmbedder, get_device, load_audio, neutralize


def process_split(embedder, data_dir, output_dir, split, batch_size):
    """Extract a whole split and save it as a single .npz file."""
    table = pd.read_csv(os.path.join(data_dir, "protocols",
                                     f"{split}_metadata.csv"))
    split_dir = os.path.join(data_dir, split)
    vectors, batch = [], []

    # Waveforms are accumulated in `batch` and embedded batch_size at a time.
    for i, filename in enumerate(table.filename, start=1):
        batch.append(neutralize(load_audio(os.path.join(split_dir, filename))))
        if len(batch) == batch_size:
            vectors.append(embedder(batch))
            batch = []
            print(f"\r{split}: {i}/{len(table)}", end="", flush=True)
    if batch:
        vectors.append(embedder(batch))

    os.makedirs(output_dir, exist_ok=True)
    output = os.path.join(output_dir, f"{split}.npz")
    np.savez_compressed(output,
                        X=np.concatenate(vectors),
                        y=table.label.to_numpy(dtype=np.int64),
                        filenames=table.filename.to_numpy())
    print(f"\r{split}: {len(table)} files -> {output}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog=__doc__.split("\n\n", 1)[1],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True,
                        help="HISPASpoof detection folder (with protocols/)")
    parser.add_argument("--output-dir", default="features",
                        help="where to write the .npz files (default: %(default)s)")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        help="splits to process (default: train val test)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="audio files per XLS-R forward pass (default: %(default)s)")
    parser.add_argument("--device", default="auto",
                        help='"auto", "cpu", "cuda" or "cuda:N" (default: auto)')
    parser.add_argument("--seed", type=int, default=1234,
                        help="seed of the neutralization noise (default: %(default)s)")
    args = parser.parse_args()

    # Hide the expected "unused weights" report of loading XLS-R without its
    # pre-training heads.
    hf_logging.set_verbosity_error()
    torch.manual_seed(args.seed)
    embedder = XLSREmbedder(get_device(args.device))
    for split in args.splits:
        process_split(embedder, args.data_dir, args.output_dir, split,
                      args.batch_size)


if __name__ == "__main__":
    main()
