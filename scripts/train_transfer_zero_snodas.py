#!/usr/bin/env python

import argparse
import glob
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import deep_snow.models
from deep_snow.dataset import norm_dict, random_transform
from deep_snow.utils import calc_norm


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
    parser.add_argument(
        "--train-glob",
        help="Glob for training NetCDF files. Defaults to all local .nc files in this project.",
    )
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
    parser.add_argument("--crop-size", type=int, default=256, help="Square patch size for training.")
    parser.add_argument("--samples-per-file", type=int, default=64)
    parser.add_argument("--val-samples-per-file", type=int, default=16)
    return parser.parse_args()


def expand_paths(pattern):
    paths = sorted(glob.glob(pattern, recursive=True))
    return [path for path in paths if os.path.isfile(path)]


def discover_local_netcdfs():
    ignored_dirs = {".git", "__pycache__"}
    paths = []
    for path in REPO_ROOT.rglob("*.nc"):
        if any(part in ignored_dirs for part in path.parts):
            continue
        paths.append(str(path))
    return sorted(paths)


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


class PatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        paths,
        samples_per_file,
        crop_size,
        augment,
        deterministic=False,
        seed=0,
        split="all",
        val_fraction=0.2,
    ):
        self.paths = paths
        self.samples_per_file = samples_per_file
        self.crop_size = crop_size
        self.augment = augment
        self.deterministic = deterministic
        self.seed = seed
        self.split = split
        self.val_fraction = val_fraction
        self.crop_windows = self.build_crop_windows()

    def build_crop_windows(self):
        crop_windows = []
        rng = random.Random(self.seed)
        for path in self.paths:
            with xr.open_dataset(path) as ds:
                height = ds.sizes["y"]
                width = ds.sizes["x"]

            if height < self.crop_size or width < self.crop_size:
                raise ValueError(
                    f"{path} is {height}x{width}, smaller than crop size "
                    f"{self.crop_size}. Use a smaller --crop-size."
                )

            step = max(1, self.crop_size // 2)
            y_starts = list(range(0, height - self.crop_size + 1, step))
            x_starts = list(range(0, width - self.crop_size + 1, step))
            if y_starts[-1] != height - self.crop_size:
                y_starts.append(height - self.crop_size)
            if x_starts[-1] != width - self.crop_size:
                x_starts.append(width - self.crop_size)

            windows = [(y0, x0) for y0 in y_starts for x0 in x_starts]
            rng.shuffle(windows)
            val_count = max(1, int(round(len(windows) * self.val_fraction)))

            if self.split == "train":
                windows = windows[val_count:]
            elif self.split == "val":
                windows = windows[:val_count]
            elif self.split != "all":
                raise ValueError(f"Unknown split: {self.split}")

            if not windows:
                raise ValueError(f"No crop windows available for {path} split={self.split}.")

            crop_windows.append(windows)
        return crop_windows

    def __len__(self):
        return len(self.paths) * self.samples_per_file

    def __getitem__(self, idx):
        file_idx = idx // self.samples_per_file
        tensors = self.load_tensors(self.paths[file_idx])

        if self.deterministic:
            window_idx = idx % len(self.crop_windows[file_idx])
        else:
            window_idx = random.randrange(len(self.crop_windows[file_idx]))
        y0, x0 = self.crop_windows[file_idx][window_idx]

        y1 = y0 + self.crop_size
        x1 = x0 + self.crop_size
        tensors = [tensor[:, y0:y1, x0:x1] for tensor in tensors]

        if self.augment:
            randoms = [random.random(), random.random(), random.randint(0, 3)]
            tensors = [random_transform(tensor, randoms) for tensor in tensors]

        return tuple(tensors)

    def load_tensors(self, path):
        with xr.open_dataset(path) as ds:
            return [self.get_channel(ds, channel) for channel in INPUT_CHANNELS + [TARGET_CHANNEL]]

    def get_channel(self, ds, channel):
        if channel == "snodas_sd":
            tensor = self.get_var(ds, "snodas_sd")
            tensor = torch.clamp(calc_norm(tensor, norm_dict["aso_sd"]), 0, 1)
        elif channel == "blue":
            tensor = torch.clamp(calc_norm(self.get_var(ds, "B02"), norm_dict["blue"]), 0, 1)
        elif channel == "swir1":
            tensor = torch.clamp(calc_norm(self.get_var(ds, "B11"), norm_dict["swir1"]), 0, 1)
        elif channel == "ndsi":
            tensor = torch.clamp(calc_norm(self.get_var(ds, "ndsi"), [-1, 1]), 0, 1)
        elif channel == "elevation":
            tensor = torch.clamp(calc_norm(self.get_var(ds, "elevation"), norm_dict["elevation"]), 0, 1)
        elif channel == "northness":
            tensor = torch.clamp(calc_norm(self.get_var(ds, "northness"), [0, 1]), 0, 1)
        elif channel == "slope":
            tensor = torch.clamp(calc_norm(self.get_var(ds, "slope"), norm_dict["slope"]), 0, 1)
        elif channel == "curvature":
            tensor = torch.clamp(calc_norm(self.get_var(ds, "curvature"), norm_dict["curvature"]), 0, 1)
        elif channel == "dowy":
            tensor = torch.clamp(calc_norm(self.get_var(ds, "dowy"), [0, 365]), 0, 1)
        elif channel == "delta_cr":
            tensor = torch.clamp(calc_norm(self.get_var(ds, "delta_cr"), norm_dict["delta_cr"]), 0, 1)
        elif channel == "fcf":
            tensor = torch.clamp(self.get_var(ds, "fcf"), 0, 1)
        elif channel == TARGET_CHANNEL:
            if TARGET_CHANNEL not in ds:
                raise ValueError(f"NetCDF does not contain {TARGET_CHANNEL}.")
            tensor = torch.clamp(calc_norm(self.get_var(ds, TARGET_CHANNEL), norm_dict["aso_sd"]), 0, 1)
        else:
            raise ValueError(f"Unsupported channel: {channel}")

        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)
        return tensor[None, :, :]

    @staticmethod
    def get_var(ds, name):
        return torch.from_numpy(np.float32(ds[name].values))


def make_loader(
    paths,
    batch_size,
    augment,
    num_workers,
    samples_per_file,
    crop_size,
    seed,
    split="all",
    val_fraction=0.2,
):
    dataset = PatchDataset(
        paths,
        samples_per_file=samples_per_file,
        crop_size=crop_size,
        augment=augment,
        deterministic=not augment,
        seed=seed,
        split=split,
        val_fraction=val_fraction,
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

    train_paths = expand_paths(args.train_glob) if args.train_glob else discover_local_netcdfs()
    if not train_paths:
        raise FileNotFoundError("No .nc files were found in this project folder.")

    if args.val_glob:
        val_paths = expand_paths(args.val_glob)
        if not val_paths:
            raise FileNotFoundError(f"No validation files matched: {args.val_glob}")
    else:
        if len(train_paths) > 1:
            train_paths, val_paths = split_train_val(train_paths, args.val_fraction, args.seed)
        else:
            val_paths = list(train_paths)

    print(f"Device: {device}")
    print(f"Training files: {len(train_paths)}")
    for path in train_paths:
        print(f"  train: {path}")
    print(f"Validation files: {len(val_paths)}")
    for path in val_paths:
        print(f"  val:   {path}")
    print(f"Zeroed input channel: snodas_sd")
    single_file_split = len(train_paths) == 1 and train_paths == val_paths
    if single_file_split:
        print(
            "Warning: only one local NetCDF was found, so validation uses a held-out "
            f"{args.val_fraction:.0%} split of crop windows from the same file."
        )

    train_loader = make_loader(
        train_paths,
        batch_size=args.batch_size,
        augment=not args.no_augment,
        num_workers=args.num_workers,
        samples_per_file=args.samples_per_file,
        crop_size=args.crop_size,
        seed=args.seed,
        split="train" if single_file_split else "all",
        val_fraction=args.val_fraction,
    )
    val_loader = make_loader(
        val_paths,
        batch_size=args.batch_size,
        augment=False,
        num_workers=args.num_workers,
        samples_per_file=args.val_samples_per_file,
        crop_size=args.crop_size,
        seed=args.seed + 10000,
        split="val" if single_file_split else "all",
        val_fraction=args.val_fraction,
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
