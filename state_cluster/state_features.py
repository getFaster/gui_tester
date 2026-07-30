"""Extract native-resolution visual features for DroidBot observations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import torch
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rgb_to_grayscale, resize

from element_finder.dinov3 import DINOv3FeatureExtractor
from .state_dataset import StateDataset
from .state_deduplicate import load_deduplication_groups


def difference_hash(image: torch.Tensor) -> str:
    """Return a 64-bit difference hash without altering the stored screenshot."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected an RGB CHW image, got {tuple(image.shape)}")
    grayscale = rgb_to_grayscale(image)
    thumbnail = resize(
        grayscale,
        [8, 9],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    bits = (thumbnail[:, :, 1:] >= thumbnail[:, :, :-1]).reshape(-1)
    value = 0
    for bit in bits.tolist():
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def hamming_distance(first_hash: str, second_hash: str) -> int:
    """Return the number of differing bits between equal-width hex hashes."""
    if len(first_hash) != len(second_hash):
        raise ValueError("Perceptual hashes must have equal widths")
    return (int(first_hash, 16) ^ int(second_hash, 16)).bit_count()


def _atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    torch.save(value, temporary_path)
    os.replace(temporary_path, path)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def extract_perceptual_hashes(dataset: StateDataset, output_dir: Path) -> Path:
    output_path = output_dir / "perceptual_hashes.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output:
        for observation in dataset.observations:
            observation_id = observation["observation_id"]
            image = read_image(
                str(dataset.screenshot_path(observation_id)),
                mode=ImageReadMode.RGB,
            )
            record = {
                "observation_id": observation_id,
                "extractor": "difference_hash_64",
                "feature": difference_hash(image),
            }
            output.write(json.dumps(record) + "\n")
    os.replace(temporary_path, output_path)
    return output_path


@torch.no_grad()
def extract_dino_features(
    dataset: StateDataset,
    output_dir: Path,
    *,
    grounding_checkpoint: Path | None = None,
    observation_ids: Sequence[str] | None = None,
) -> None:
    dino = DINOv3FeatureExtractor()
    element_finder = None
    extractor_name = "dino_global"
    if grounding_checkpoint is not None:
        from element_finder.pipeline import load_element_finder

        element_finder = load_element_finder(
            grounding_checkpoint,
            dino.model.device,
        )
        extractor_name = "grounding"

    feature_dir = output_dir / extractor_name
    selected_ids = (
        list(observation_ids)
        if observation_ids is not None
        else [observation["observation_id"] for observation in dataset.observations]
    )
    for observation_id in selected_ids:
        image = read_image(
            str(dataset.screenshot_path(observation_id)),
            mode=ImageReadMode.RGB,
        )
        height, width = image.shape[-2:]
        if height % dino.patch_size or width % dino.patch_size:
            raise ValueError(
                f"{observation_id} has native size {(height, width)}, which is "
                f"not divisible by DINO patch size {dino.patch_size}"
            )
        features = dino(image)
        patch_features = features[:, dino.num_special_tokens :]
        payload: dict[str, object] = {
            "observation_id": observation_id,
            "image_size": (height, width),
            "patch_grid": (height // dino.patch_size, width // dino.patch_size),
            "global_embedding": patch_features.mean(dim=1).squeeze(0).cpu(),
        }
        if element_finder is not None:
            grounded_features = element_finder.model(features)
            payload["patch_features"] = (
                grounded_features[:, element_finder.num_special_tokens :]
                .squeeze(0)
                .cpu()
            )
            payload["element_logits"] = element_finder(features).squeeze(0).cpu()
        _atomic_torch_save(payload, feature_dir / f"{observation_id}.pt")

    _write_json(
        feature_dir / "manifest.json",
        {
            "run_id": dataset.run_id,
            "extractor": extractor_name,
            "model_name": getattr(
                dino.processor,
                "name_or_path",
                "facebook/dinov3-vits16-pretrain-lvd1689m",
            ),
            "grounding_checkpoint": (
                str(grounding_checkpoint.resolve())
                if grounding_checkpoint is not None
                else None
            ),
            "native_resolution": True,
            "patch_size": dino.patch_size,
            "observation_count": len(selected_ids),
            "dataset_observation_count": len(dataset.observations),
        },
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("state_features"))
    parser.add_argument(
        "--extractor",
        choices=("perceptual", "dino", "grounding"),
        default="perceptual",
    )
    parser.add_argument(
        "--grounding-checkpoint",
        type=Path,
        default=Path("checkpoints/element_finder/best.pt"),
    )
    parser.add_argument(
        "--deduplication",
        type=Path,
        help=(
            "Extract only the first observation from every validated "
            "exact-screenshot group."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset = StateDataset.load(args.run_dir)
    output_dir = args.output_dir / dataset.run_id
    selected_ids = None
    if args.deduplication is not None:
        groups = load_deduplication_groups(dataset, args.deduplication)
        selected_ids = [group["observation_ids"][0] for group in groups]
    if args.extractor == "perceptual":
        if selected_ids is not None:
            raise ValueError(
                "--deduplication currently supports dino and grounding "
                "feature extraction"
            )
        output_path = extract_perceptual_hashes(dataset, output_dir)
        print(f"Wrote {output_path}")
        return
    checkpoint = args.grounding_checkpoint if args.extractor == "grounding" else None
    extract_dino_features(
        dataset,
        output_dir,
        grounding_checkpoint=checkpoint,
        observation_ids=selected_ids,
    )
    observation_count = (
        len(selected_ids) if selected_ids is not None else len(dataset.observations)
    )
    print(
        f"Wrote {args.extractor} features for {observation_count} observations "
        f"below {output_dir}"
    )


if __name__ == "__main__":
    main()
