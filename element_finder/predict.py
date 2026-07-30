"""Use the patch-level clickable/scrollable element classifier."""

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
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .amex_dataset import AmexDataset, ProccessedDataset
from .element_finder import ElementFinder
from .train import parse_args, validate


@torch.no_grad()
def collect_predictions(
    model: ElementFinder,
    val_loader: DataLoader,
    device: torch.device,
    sub_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect patch probabilities and targets without padding ragged samples."""
    probabilities: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    model.eval()

    for features, targets in tqdm(val_loader, desc="Predictions", leave=False):
        groups: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        for feature, target in zip(features, targets, strict=True):
            groups.setdefault(feature.shape[0], []).append((feature, target))

        for group in groups.values():
            for start in range(0, len(group), sub_batch_size):
                chunk = group[start : start + sub_batch_size]
                feature_chunk, target_chunk = zip(*chunk, strict=True)
                stacked_features = torch.stack(feature_chunk).to(device)
                stacked_targets = torch.stack(target_chunk)
                patch_logits = model(stacked_features)
                if patch_logits.shape != stacked_targets.shape:
                    raise ValueError(
                        "Patch logits and targets must have the same shape, got "
                        f"{tuple(patch_logits.shape)} and "
                        f"{tuple(stacked_targets.shape)}."
                    )
                probabilities.append(torch.sigmoid(patch_logits).cpu().reshape(-1, 2))
                all_targets.append(stacked_targets.cpu().reshape(-1, 2))

    if not probabilities:
        raise ValueError("The validation loader did not produce any samples.")
    return torch.cat(probabilities), torch.cat(all_targets)


def best_f1_operating_point(
    probabilities: torch.Tensor, targets: torch.Tensor
) -> tuple[float, float, float, float]:
    """Return the exact threshold, precision, recall, and F1 optimum."""
    probabilities = probabilities.flatten()
    targets = targets.flatten().to(device=probabilities.device, dtype=torch.bool)
    if probabilities.numel() == 0:
        raise ValueError("probabilities and targets must not be empty.")
    if probabilities.shape != targets.shape:
        raise ValueError(
            "probabilities and targets must have the same number of elements."
        )

    sorted_probabilities, order = probabilities.sort(descending=True)
    sorted_targets = targets[order]
    true_positives = sorted_targets.cumsum(0)
    predicted_positives = torch.arange(1, targets.numel() + 1, device=targets.device)
    total_positives = sorted_targets.sum()
    precision = true_positives / predicted_positives
    recall = true_positives / total_positives.clamp_min(1)
    f1 = (
        2
        * precision
        * recall
        / (precision + recall).clamp_min(torch.finfo(precision.dtype).eps)
    )

    # A threshold cannot split predictions with identical scores. Evaluate only
    # the final item in each tied group, where every value >= the threshold has
    # been included.
    distinct_threshold = torch.ones_like(sorted_probabilities, dtype=torch.bool)
    distinct_threshold[:-1] = sorted_probabilities[:-1] != sorted_probabilities[1:]
    candidate_indices = distinct_threshold.nonzero(as_tuple=True)[0]
    best_index = candidate_indices[f1[candidate_indices].argmax()]
    return (
        sorted_probabilities[best_index].item(),
        precision[best_index].item(),
        recall[best_index].item(),
        f1[best_index].item(),
    )


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
    _, val_dataset = random_split(
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
    val_loader = DataLoader(val_dataset, shuffle=True, **loader_options)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ElementFinder().to(device)
    # model.compile(mode="reduce-overhead")

    print(f"Validating on {len(val_dataset)} samples.")

    checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    print("Newly initialized parameters:", missing)
    print("Unexpected parameters:", unexpected)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    validation_loss = validate(model, val_loader, loss_fn, device, args)
    print(f"Validation loss: {validation_loss:.6f}")

    probabilities, targets = collect_predictions(
        model, val_loader, device, args.loss_sub_batch_size
    )
    with SummaryWriter(log_dir=str(args.log_dir)) as writer:
        writer.add_pr_curve(
            "precision_recall/all",
            targets.flatten(),
            probabilities.flatten(),
        )
        for class_index, class_name in enumerate(("clickable", "scrollable")):
            writer.add_pr_curve(
                f"precision_recall/{class_name}",
                targets[:, class_index],
                probabilities[:, class_index],
            )
        writer.flush()

    threshold, precision, recall, f1 = best_f1_operating_point(probabilities, targets)
    print(
        f"Best F1 threshold: {threshold:.6f} "
        f"(F1: {f1:.6f}, precision: {precision:.6f}, recall: {recall:.6f})"
    )
    print(f"Precision-recall curves written to {args.log_dir}.")
    checkpoint["threshold"] = threshold
    torch.save(checkpoint, args.resume)


if __name__ == "__main__":
    main()
