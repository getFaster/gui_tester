"""Visualize patch-level element predictions on screenshots."""

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Sequence

# PyTorch's bundled Inductor kernels are UTF-8 text. On Windows, restart with
# UTF-8 mode before importing PyTorch, matching the training entry point.
if __name__ == "__main__" and not sys.flags.utf8_mode:
    utf8_process = subprocess.run(
        [sys.executable, "-X", "utf8", *sys.argv], check=False
    )
    raise SystemExit(utf8_process.returncode)

import torch
import torch.nn.functional as F
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from torchvision.utils import save_image

from dinov3 import DINOv3FeatureExtractor
from element_finder import ElementFinder
from network import AttentionLayer


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
CLASS_NAMES = ("clickable", "scrollable")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("image_data"),
        help="Directory containing screenshots (default: image_data).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("visualizations"),
        help="Directory in which to write heatmaps (default: visualizations).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=Path("checkpoints/element_finder/best.pt"),
        help="ElementFinder checkpoint to load.",
    )
    parser.add_argument(
        "--resize-scale",
        type=float,
        default=0.25,
        help="Scale applied to both image dimensions before inference.",
    )
    parser.add_argument(
        "--overlay-opacity",
        type=float,
        default=0.75,
        help="Maximum opacity of the blue probability overlay.",
    )
    return parser.parse_args(argv)


def load_images_from_directory(input_dir: Path) -> list[Path]:
    """Return supported image files below ``input_dir`` in stable order."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    image_paths = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise RuntimeError(f"No supported images found in {input_dir}")
    return image_paths


def resize_image(image: torch.Tensor, scale: float) -> torch.Tensor:
    """Resize a CHW image while keeping height and width ordering explicit."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected an RGB CHW image, got {tuple(image.shape)}")
    height, width = image.shape[-2:]
    resized_height = max(1, round(height * scale))
    resized_width = max(1, round(width * scale))
    return resize(
        image,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )


def render_probability_overlay(
    image: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    tile_size: int,
    opacity: float,
    color: tuple[int, int, int] = (0, 0, 255),
) -> torch.Tensor:
    """Blend a patch-probability grid over an RGB CHW image.

    ``probabilities[y, x]`` maps to the DINO patch at row ``y`` and column
    ``x``. Any partial patch strip at the bottom or right stays unchanged.
    """
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected an RGB CHW image, got {tuple(image.shape)}")
    if probabilities.ndim != 2:
        raise ValueError(
            "Expected probabilities with shape (patch_rows, patch_columns), "
            f"got {tuple(probabilities.shape)}"
        )
    if tile_size < 1:
        raise ValueError("tile_size must be positive")
    if not 0 <= opacity <= 1:
        raise ValueError("opacity must be in the interval [0, 1]")

    patch_rows, patch_columns = probabilities.shape
    covered_height = patch_rows * tile_size
    covered_width = patch_columns * tile_size
    height, width = image.shape[-2:]
    if covered_height > height or covered_width > width:
        raise ValueError(
            "Probability grid is larger than the image: "
            f"grid covers {(covered_height, covered_width)}, image is {(height, width)}"
        )

    output = image.to(dtype=torch.float32).clone()
    if image.dtype == torch.uint8:
        output /= 255
    alpha = F.interpolate(
        probabilities.clamp(0, 1)[None, None].to(dtype=torch.float32),
        size=(covered_height, covered_width),
        mode="nearest",
    ).squeeze(0)
    alpha *= opacity
    overlay_color = torch.tensor(color, dtype=torch.float32).view(3, 1, 1) / 255
    covered_image = output[:, :covered_height, :covered_width]
    output[:, :covered_height, :covered_width] = (
        covered_image * (1 - alpha) + overlay_color * alpha
    )
    return output.clamp(0, 1)


def eager_softcap_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """Evaluate the model's soft-capped attention without compiling a kernel."""
    scores = query @ key.transpose(-2, -1)
    scores = torch.tanh(scores / 30) * 30
    attention = torch.softmax(scores, dim=-1)
    return attention @ value


def load_element_finder(checkpoint_path: Path, device: torch.device) -> ElementFinder:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Checkpoint does not contain a model state dictionary: {checkpoint_path}"
        )
    model = ElementFinder().to(device)
    model.load_state_dict(state_dict)
    for module in model.modules():
        if isinstance(module, AttentionLayer):
            module.MHA = eager_softcap_attention
    return model.eval()


@torch.no_grad()
def visualize_image(
    image: torch.Tensor,
    dino_model: DINOv3FeatureExtractor,
    element_finder: ElementFinder,
    *,
    opacity: float,
) -> dict[str, torch.Tensor]:
    """Return one blue probability overlay for each predicted element class."""
    tile_size = dino_model.patch_size
    height, width = image.shape[-2:]
    patch_rows = height // tile_size
    patch_columns = width // tile_size
    if patch_rows == 0 or patch_columns == 0:
        raise ValueError(
            f"Resized image {(height, width)} is smaller than one {tile_size}x{tile_size} patch"
        )

    features = dino_model(image)
    logits = element_finder(features)
    probabilities = torch.sigmoid(logits).squeeze(0).cpu()
    expected_shape = (patch_rows * patch_columns, len(CLASS_NAMES))
    if probabilities.shape != expected_shape:
        raise ValueError(
            "Patch predictions do not match the resized image grid: "
            f"got {tuple(probabilities.shape)}, expected {expected_shape}. "
            "Check the DINO patch size and number of special tokens."
        )

    probability_grid = probabilities.reshape(
        patch_rows, patch_columns, len(CLASS_NAMES)
    )
    return {
        class_name: render_probability_overlay(
            image.cpu(),
            probability_grid[:, :, class_index],
            tile_size=tile_size,
            opacity=opacity,
        )
        for class_index, class_name in enumerate(CLASS_NAMES)
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.resize_scale <= 0:
        raise ValueError("resize-scale must be positive")
    if not 0 <= args.overlay_opacity <= 1:
        raise ValueError("overlay-opacity must be in the interval [0, 1]")

    image_paths = load_images_from_directory(args.input_dir)
    dino_model = DINOv3FeatureExtractor()
    device = dino_model.model.device
    element_finder = load_element_finder(args.resume, device)

    written = 0
    for image_path in image_paths:
        image = read_image(str(image_path), mode=ImageReadMode.RGB)
        image = resize_image(image, args.resize_scale)
        overlays = visualize_image(
            image,
            dino_model,
            element_finder,
            opacity=args.overlay_opacity,
        )
        relative_path = image_path.relative_to(args.input_dir)
        output_parent = args.output_dir / relative_path.parent
        output_parent.mkdir(parents=True, exist_ok=True)
        for class_name, overlay in overlays.items():
            output_path = output_parent / f"{relative_path.stem}_{class_name}.png"
            save_image(overlay, output_path)
            written += 1

    print(f"Wrote {written} visualization(s) to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
