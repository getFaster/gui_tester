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
        default=8,
        help="Maximum number of equal-length feature sequences forwarded together.",
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
    """Compute one equally weighted loss per screenshot in a ragged batch.

    Samples are grouped by token count so every model forward receives a dense,
    unpadded tensor.
    """
    if len(features) != len(targets):
        raise ValueError(
            "features and targets must contain the same number of samples, got "
            f"{len(features)} and {len(targets)}."
        )
    if not features:
        raise ValueError("features and targets must not be empty.")
    if sub_batch_size < 1:
        raise ValueError("sub_batch_size must be positive.")
    if not isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
        raise TypeError("batch_loss requires BCEWithLogitsLoss.")
    if loss_fn.reduction != "mean":
        raise ValueError("batch_loss requires BCEWithLogitsLoss reduction='mean'.")

    num_special_tokens = model.num_special_tokens
    batch_size = len(features)
    embedding_dim = features[0].shape[-1]
    groups: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    for sample_index, (feature, target) in enumerate(
        zip(features, targets, strict=True)
    ):
        if feature.ndim != 2 or feature.shape[-1] != embedding_dim:
            raise ValueError(
                "Every feature must have shape (tokens, embedding_dim); "
                f"sample {sample_index} has shape {tuple(feature.shape)}."
            )
        expected_targets = feature.shape[0] - num_special_tokens
        if target.ndim != 2 or target.shape[0] != expected_targets:
            raise ValueError(
                "Each target must have one row per patch token; "
                f"sample {sample_index} has {feature.shape[0]} feature tokens "
                f"and target shape {tuple(target.shape)}, expected "
                f"({expected_targets}, classes)."
            )
        groups.setdefault(feature.shape[0], []).append((feature, target))

    cumulative_loss = torch.zeros((), device=device)
    for group in groups.values():
        for start in range(0, len(group), sub_batch_size):
            chunk = group[start : start + sub_batch_size]
            feature_chunk, target_chunk = zip(*chunk, strict=True)
            stacked_features = torch.stack(feature_chunk).to(device)
            stacked_targets = torch.stack(target_chunk).to(device)
            torch.compiler.cudagraph_mark_step_begin()
            patch_logits = model(stacked_features)

            if patch_logits.shape != stacked_targets.shape:
                raise ValueError(
                    "Patch logits and targets must have the same shape, got "
                    f"{tuple(patch_logits.shape)} and "
                    f"{tuple(stacked_targets.shape)}."
                )
            element_loss = F.binary_cross_entropy_with_logits(
                patch_logits,
                stacked_targets,
                weight=loss_fn.weight,
                pos_weight=loss_fn.pos_weight,
                reduction="none",
            )
            sample_loss = element_loss.mean(dim=(1, 2))
            chunk_loss = sample_loss.sum() / batch_size
            if backward:
                chunk_loss.backward()
            cumulative_loss += sample_loss.detach().sum()

    return (cumulative_loss / batch_size).item()


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


def load_checkpoint(model, optim_muon, optim_adamw, args, device):
    checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optim_muon.load_state_dict(checkpoint["optim_muon_state_dict"])
    optim_adamw.load_state_dict(checkpoint["optim_adamw_state_dict"])
    start_epoch = int(checkpoint["epoch"]) + 1
    print(f"Resumed from {args.resume} at epoch {start_epoch + 1}.")
    return checkpoint


def changing_load(model, optim_muon, optim_adamw, args, device):
    checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
    old_state = checkpoint["model_state_dict"]
    new_state = model.state_dict()

    # Load every unchanged parameter with a matching shape.
    compatible_state = {
        name: value
        for name, value in old_state.items()
        if name in new_state and value.shape == new_state[name].shape
    }

    for name in ["u", "gate", "out"]:
        compatible_state[f"head.3.{name}.weight"] = old_state[f"head.1.{name}.weight"]
        compatible_state[f"head.3.{name}.bias"] = old_state[f"head.1.{name}.bias"]

    compatible_state["head.4.weight"] = old_state["head.2.weight"]
    compatible_state["head.4.bias"] = old_state["head.2.bias"]

    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    print("Newly initialized parameters:", missing)
    print("Unexpected parameters:", unexpected)

    model_state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(model_state_dict, strict=False)
    # optim_muon.load_state_dict(checkpoint["optim_muon_state_dict"])
    # optim_adamw.load_state_dict(checkpoint["optim_adamw_state_dict"])
    start_epoch = int(checkpoint["epoch"]) + 1
    print(f"Resumed from {args.resume} at epoch {start_epoch + 1}.")
    return checkpoint


@torch.no_grad()
def validate(model, val_loader, loss_fn, device, args):
    model.eval()
    val_total = 0.0
    sample_count = 0
    for images, masks in tqdm(val_loader, desc="Validation", leave=False):
        tmp_loss = batch_loss(
            model,
            images,
            masks,
            loss_fn,
            device,
            args.loss_sub_batch_size,
            backward=False,
        )
        val_total += tmp_loss * len(images)  # weighted by batch size
        sample_count += len(images)
    val_loss = val_total / sample_count
    return val_loss


def need_new_optimizer(checkpoint, adamw_hparams, muon_hparams) -> bool:
    for hname, hparam in adamw_hparams.items():
        if checkpoint["optim_adamw_state_dict"]["param_groups"][0][hname] != hparam:
            return True
    for hname, hparam in muon_hparams.items():
        if checkpoint["optim_muon_state_dict"]["param_groups"][0][hname] != hparam:
            return True
    return False


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
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=True, **loader_options)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ElementFinder().to(device)
    # model.compile(mode="reduce-overhead")
    muon_params = [p for p in model.parameters() if p.ndim == 2]
    other_params = [p for p in model.parameters() if p.ndim != 2]
    muon_hparams: dict[str, Any] = {
        "lr": 1e-5,
        "weight_decay": 0.04,
        "momentum": 0.95,
    }
    adamw_hparams = {
        "lr": 5e-5,
        "weight_decay": 0.04,
        "amsgrad": True,
    }

    optim_muon = torch.optim.Muon(muon_params, **muon_hparams)
    optim_adamw = torch.optim.AdamW(other_params, **adamw_hparams)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    print(
        f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples."
    )
    start_epoch, best_val_loss = 0, float("inf")
    if args.resume:
        changing_architecture = True
        if changing_architecture:
            checkpoint = changing_load(model, optim_muon, optim_adamw, args, device)
            start_epoch = int(checkpoint["epoch"]) + 1
            best_val_loss = float(checkpoint["best_val_loss"])
        else:
            checkpoint = load_checkpoint(model, optim_muon, optim_adamw, args, device)
            if need_new_optimizer(checkpoint, adamw_hparams, muon_hparams):
                print("Optimizer hyperparameters changed: using a new optimizer.")
                optim_muon = torch.optim.Muon(muon_params, **muon_hparams)
                optim_adamw = torch.optim.AdamW(other_params, **adamw_hparams)
            start_epoch = int(checkpoint["epoch"]) + 1
            best_val_loss = float(checkpoint["best_val_loss"])

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(args.log_dir)) as writer:
        hparams = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "loss_sub_batch_size": args.loss_sub_batch_size,
            "val_size": args.val_size,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "muon_lr": optim_muon.param_groups[0]["lr"],
            "muon_weight_decay": optim_muon.param_groups[0]["weight_decay"],
            "muon_momentum": optim_muon.param_groups[0]["momentum"],
            "adamw_lr": optim_adamw.param_groups[0]["lr"],
            "adamw_weight_decay": optim_adamw.param_groups[0]["weight_decay"],
            "adamw_amsgrad": optim_adamw.param_groups[0]["amsgrad"],
        }

        writer.add_hparams(hparams, {})

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
            val_loss = validate(model, val_loader, loss_fn, device, args)
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
            hparams,
            {"hparam/train_loss": train_loss, "hparam/validation_loss": val_loss},
        )


if __name__ == "__main__":
    main()
