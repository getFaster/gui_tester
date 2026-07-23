"""Train the patch-level clickable/scrollable element classifier."""

import argparse
from pathlib import Path
import subprocess
import sys

# TorchInductor's bundled kernels are UTF-8 text. On Windows, Python otherwise
# opens them with the active ANSI code page (CP950 on Traditional Chinese
# systems), which can crash while constructing a torch.compile() wrapper.
# UTF-8 mode is fixed at interpreter startup, so restart this script once with
# Python's documented command-line setting before importing PyTorch.
if __name__ == "__main__" and not sys.flags.utf8_mode:
    utf8_process = subprocess.run(
        [sys.executable, "-X", "utf8", *sys.argv], check=False
    )
    raise SystemExit(utf8_process.returncode)

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from amex_dataset import AmexDataset, ProccessedDataset
from element_finder import ElementFinder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshot-dir", default="screenshot")
    parser.add_argument("--annotation-dir", default="element_anno")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--loss-sub-batch-size",
        type=int,
        default=4,
        help="Number of ragged feature sequences padded and forwarded together.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=1000,
        help="Number of samples reserved for validation (default: 1000).",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=Path, default=Path("runs/element_finder"))
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints/element_finder")
    )
    parser.add_argument(
        "--resume", type=Path, default=None, help="Checkpoint to resume from."
    )
    return parser.parse_args()


def batch_loss(
    model: ElementFinder,
    features: list[torch.Tensor],
    targets: list[torch.Tensor],
    loss_fn: torch.nn.Module,
    device: torch.device,
    sub_batch_size: int,
    backward: bool,
) -> float:
    """Compute one equally weighted loss per screenshot in a ragged batch."""
    if len(features) != len(targets):
        raise ValueError(
            "features and targets must contain the same number of samples, got "
            f"{len(features)} and {len(targets)}."
        )
    if sub_batch_size < 1:
        raise ValueError("sub_batch_size must be positive.")
    if not isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
        raise TypeError("batch_loss requires BCEWithLogitsLoss.")

    num_special_tokens = model.num_special_tokens
    cumulative_loss = 0.0
    batch_size = len(features)
    for start in range(0, batch_size, sub_batch_size):
        feature_chunk = features[start : start + sub_batch_size]
        target_chunk = targets[start : start + sub_batch_size]
        embedding_dim = feature_chunk[0].shape[-1]

        feature_lengths = torch.tensor(
            [feature.shape[0] for feature in feature_chunk],
            dtype=torch.long,
            device=device,
        )
        padded_features = pad_sequence(
            [feature.to(device) for feature in feature_chunk], batch_first=True
        )
        padded_targets = pad_sequence(
            [target.to(device) for target in target_chunk], batch_first=True
        )
        patch_logits = model(padded_features)

        # TODO: calculate chunk_loss
        if backward:
            chunk_loss.backward()
        cumulative_loss += sample_loss.detach().sum().item()

    return cumulative_loss / batch_size


def make_checkpoint(
    epoch: int,
    model: ElementFinder,
    optim_muon: torch.optim.Optimizer,
    optim_adamw: torch.optim.Optimizer,
    best_val_loss: float,
) -> dict[str, object]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optim_muon_state_dict": optim_muon.state_dict(),
        "optim_adamw_state_dict": optim_adamw.state_dict(),
        "best_val_loss": best_val_loss,
    }


def main() -> None:
    args = parse_args()
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.loss_sub_batch_size < 1
        or args.val_size < 1
    ):
        raise ValueError(
            "epochs, batch-size, loss-sub-batch-size, and val-size must all be "
            "positive."
        )

    # dataset = AmexDataset(args.screenshot_dir, args.annotation_dir)
    dataset = ProccessedDataset("feature", "target")
    if args.val_size >= len(dataset):
        raise ValueError("val-size must be smaller than the number of dataset samples.")
    train_dataset, val_dataset = random_split(
        dataset,
        [len(dataset) - args.val_size, args.val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader_options = {
        "batch_size": args.batch_size,
        "collate_fn": AmexDataset.collate_fn,
        "num_workers": args.num_workers,
        "pin_memory": False,  # somehow the way we load the pt files is not compatible with pin_memory
        # RuntimeError: cannot pin 'torch.cuda.FloatTensor' only dense CPU tensors can be pinned
        # "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ElementFinder().to(device)
    # model.compile(mode="reduce-overhead")
    muon_params = [p for p in model.parameters() if p.ndim == 2]
    other_params = [p for p in model.parameters() if p.ndim != 2]
    optim_muon = torch.optim.Muon(muon_params, lr=0.02, momentum=0.95)
    optim_adamw = torch.optim.AdamW(other_params, lr=3e-4, weight_decay=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    print(
        f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples."
    )
    start_epoch, best_val_loss = 0, float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optim_muon.load_state_dict(checkpoint["optim_muon_state_dict"])
        optim_adamw.load_state_dict(checkpoint["optim_adamw_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint["best_val_loss"])
        print(f"Resumed from {args.resume} at epoch {start_epoch + 1}.")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(args.log_dir)) as writer:
        train_loss, val_loss = None, None
        for epoch in range(start_epoch, args.epochs):
            model.train()
            train_total = 0.0
            progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
            for images, masks in progress:
                optim_muon.zero_grad(set_to_none=True)
                optim_adamw.zero_grad(set_to_none=True)
                loss = batch_loss(
                    model,
                    images,
                    masks,
                    loss_fn,
                    device,
                    args.loss_sub_batch_size,
                    backward=True,
                )
                optim_muon.step()
                optim_adamw.step()
                train_total += loss
                progress.set_postfix(train_loss=f"{loss:.4f}")
                writer.add_scalar(
                    "loss/train",
                    loss,
                    epoch * len(train_loader) + progress.n,
                )
            train_loss = train_total / len(train_loader)
            model.eval()
            val_total = 0.0
            with torch.no_grad():
                for images, masks in tqdm(val_loader, desc="Validation", leave=False):
                    val_total += batch_loss(
                        model,
                        images,
                        masks,
                        loss_fn,
                        device,
                        args.loss_sub_batch_size,
                        backward=False,
                    )
            val_loss = val_total / len(val_loader)
            writer.add_scalar("loss/validation", val_loss, epoch + 1)
            writer.flush()

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            checkpoint = make_checkpoint(
                epoch, model, optim_muon, optim_adamw, best_val_loss
            )
            torch.save(checkpoint, args.checkpoint_dir / "last.pt")
            if is_best:
                torch.save(checkpoint, args.checkpoint_dir / "best.pt")
            print(
                f"Epoch {epoch + 1}/{args.epochs}: "
                f"train loss {train_loss:.5f}, validation loss {val_loss:.5f}"
            )

        writer.add_hparams(
            {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "loss_sub_batch_size": args.loss_sub_batch_size,
                "val_size": args.val_size,
                "num_workers": args.num_workers,
                "seed": args.seed,
            },
            {"hparam/train_loss": train_loss, "hparam/validation_loss": val_loss},
        )


if __name__ == "__main__":
    main()
