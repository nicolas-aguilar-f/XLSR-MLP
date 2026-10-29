"""Train the MLP classifier on the neutralized XLS-R embeddings.

Architecture:  2048 -> 256 -> 2   (525,570 parameters)
Labels:        0 = bona fide (real),  1 = spoof (synthetic)

After training, the decision threshold is computed on the VALIDATION set and
stored in the checkpoint. Calibrating it on validation, and not on the set
that is later reported, avoids the optimistic bias of choosing the threshold
that minimizes the error on the evaluated data itself.

Example:
    python train_mlp.py --features-dir features --output models/my_model.pth
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve

from xlsr_mlp import (CKPT_MEAN, CKPT_MODEL, CKPT_STD, CKPT_THRESHOLD,
                      CKPT_VAL_EER, MLPClassifier, get_device)


def load_split(features_dir, split):
    d = np.load(os.path.join(features_dir, f"{split}.npz"), allow_pickle=True)
    return d["X"].astype(np.float32), d["y"].astype(np.int64)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog=__doc__.split("\n\n", 1)[1],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-dir", default="features",
                        help="folder with train.npz and val.npz (default: %(default)s)")
    parser.add_argument("--output", default="models/model.pth",
                        help="checkpoint to write (default: %(default)s)")
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto",
                        help='"auto", "cpu", "cuda" or "cuda:N" (default: auto)')
    args = parser.parse_args()

    device = get_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X, y = load_split(args.features_dir, "train")

    # Statistics computed on train only; they are stored in the checkpoint
    # because inference must apply exactly the same normalization.
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X = torch.from_numpy((X - mean) / std).to(device)
    y = torch.from_numpy(y).to(device)

    # Weights inversely proportional to class frequency: the dataset is
    # imbalanced and without them the network leans to the majority class.
    n_real = int((y == 0).sum())
    n_spoof = int((y == 1).sum())
    weights = torch.tensor([(n_real + n_spoof) / (2 * n_real),
                            (n_real + n_spoof) / (2 * n_spoof)],
                           dtype=torch.float32, device=device)

    model = MLPClassifier().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=args.epochs)

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(X), device=device)
        total_loss = 0.0

        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            if len(idx) < 2:          # BatchNorm needs at least two samples
                continue
            optimizer.zero_grad()
            loss = criterion(model(X[idx]), y[idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        scheduler.step()
        print(f"epoch {epoch}/{args.epochs}  loss = {total_loss / len(order):.4f}")

    # ---------------------------------------------------------- calibration
    X_val, y_val = load_split(args.features_dir, "val")
    X_val = torch.from_numpy((X_val - mean) / std).to(device)

    model.eval()
    with torch.no_grad():
        prob_val = torch.softmax(model(X_val), dim=1)[:, 1].cpu().numpy()

    fpr, tpr, thresholds = roc_curve(y_val, prob_val)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    threshold = float(thresholds[i])
    eer_val = (fpr[i] + fnr[i]) / 2 * 100

    print(f"\nvalidation: EER = {eer_val:.2f} %   threshold = {threshold:.6f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({CKPT_MODEL: model.state_dict(), CKPT_MEAN: mean, CKPT_STD: std,
                CKPT_THRESHOLD: threshold, CKPT_VAL_EER: eer_val},
               args.output)
    print(f"checkpoint saved to {args.output}")


if __name__ == "__main__":
    main()
