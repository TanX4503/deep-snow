#!/usr/bin/env python

import argparse
import glob
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import deep_snow.models
from deep_snow.dataset import Datasetv2


INPUT_CHANNELS = [
    "snodas_sd",
    "blue",
    "swir1",
    "ndsi",
    "elevation",
    "northness",
    "slope",
    "curvature",
    "dowy",
    "delta_cr",
    "fcf",
]

TARGET_CHANNEL = "aso_sd"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune ResDepth with the snodas_sd input channel forced to zero."
    )
    parser.add_argument("--train-glob", required=True, help="Glob for training NetCDF files.")
    parser.add_argument("--val-glob", help="Glob for validation NetCDF files.")
    parser.add_argument("--checkpoint", required=True, help="Starting 11-channel ResDepth checkpoint.")
    parser.add_argument("--out-dir", default="weights", help="Directory for fine-tuned weights.")
    parser.add_argument("--name", default="ResDepth_zero_snodas_transfer", help="Output weight name prefix.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-augment", action="store_true", help="Disable training augmentations.")
    return parser.parse_args()


def expand_paths(pattern):
    paths = sorted(glob.glob(pattern, recursive=True))
    return [path for path in paths if os.path.isfile(path)]


def split_train_val(paths, val_fraction, seed):
    paths = list(paths)
    random.Random(seed).shuffle(paths)
    if len(paths) < 2:
        return paths, []
    val_count = max(1, int(round(len(paths) * val_fraction)))
    return paths[val_count:], paths[:val_count]


def load_checkpoint(path, device):
    state_dict = torch.load(path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    first_weight = state_dict.get("encoder.0.0.0.weight")
    if first_weight is None:
        raise ValueError("Checkpoint does not look like a ResDepth state_dict.")
    checkpoint_channels = first_weight.shape[1]
    if checkpoint_channels != len(INPUT_CHANNELS):
        raise ValueError(
            f"Checkpoint has {checkpoint_channels} input channels, "
            f"but this zero-SNODAS transfer script uses {len(INPUT_CHANNELS)}."
        )
    return state_dict


def make_loader(paths, batch_size, augment, num_workers):
    selected_channels = INPUT_CHANNELS + [TARGET_CHANNEL]
    dataset = Datasetv2(
        paths,
        selected_channels,
        norm=True,
        augment=augment,
        cache_data=False,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=augment,
        num_workers=num_workers,
    )


def batch_to_inputs_target(batch, device):
    tensors = [item.to(device, non_blocking=True) for item in batch]
    inputs = torch.cat(tensors[:len(INPUT_CHANNELS)], dim=1)
    target = tensors[-1]

    snodas_idx = INPUT_CHANNELS.index("snodas_sd")
    inputs[:, snodas_idx:snodas_idx + 1, :, :] = 0

    inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1.0, neginf=0.0)
    return inputs, target


def masked_mse(pred, target):
    mask = torch.isfinite(pred) & torch.isfinite(target)
    if not mask.any():
        return None
    return ((pred[mask] - target[mask]) ** 2).mean()


def run_epoch(model, loader, optimizer, device):
    training = optimizer is not None
    model.train(training)
    losses = []

    for batch in loader:
        inputs, target = batch_to_inputs_target(batch, device)

        with torch.set_grad_enabled(training):
            pred = model(inputs)
            loss = masked_mse(pred, target)
            if loss is None:
                continue

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        losses.append(loss.detach().item())

    return float(np.mean(losses)) if losses else float("nan")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    train_paths = expand_paths(args.train_glob)
    if not train_paths:
        raise FileNotFoundError(f"No training files matched: {args.train_glob}")

    if args.val_glob:
        val_paths = expand_paths(args.val_glob)
        if not val_paths:
            raise FileNotFoundError(f"No validation files matched: {args.val_glob}")
    else:
        train_paths, val_paths = split_train_val(train_paths, args.val_fraction, args.seed)

    print(f"Device: {device}")
    print(f"Training files: {len(train_paths)}")
    print(f"Validation files: {len(val_paths)}")
    print(f"Zeroed input channel: snodas_sd")

    train_loader = make_loader(
        train_paths,
        batch_size=args.batch_size,
        augment=not args.no_augment,
        num_workers=args.num_workers,
    )
    val_loader = make_loader(
        val_paths,
        batch_size=args.batch_size,
        augment=False,
        num_workers=args.num_workers,
    ) if val_paths else None

    model = deep_snow.models.ResDepth(n_input_channels=len(INPUT_CHANNELS), depth=5)
    model.load_state_dict(load_checkpoint(args.checkpoint, device))
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / f"{args.name}_best"
    final_path = out_dir / f"{args.name}_final"

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device)
        if val_loader is not None:
            val_loss = run_epoch(model, val_loader, None, device)
            print(f"Epoch {epoch:03d}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), best_path)
                print(f"Saved best checkpoint: {best_path}")
        else:
            print(f"Epoch {epoch:03d}: train_loss={train_loss:.6f}")

    torch.save(model.state_dict(), final_path)
    print(f"Saved final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
