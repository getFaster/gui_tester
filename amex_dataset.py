"""PyTorch dataset for the screenshot subset of AMEX."""

import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize


def bboxes_to_mask(
    boxes: Sequence[Sequence[float]], image_size: tuple[int, int], tile_size: int
) -> torch.Tensor:
    """Convert bounding boxes to a boolean mask of tiles.

    Args:
        boxes: ``N x 4`` of bounding boxes in ``(x1, y1, x2, y2)`` format.
        image_size: Tuple of ``(height, width)`` of the image.
        tile_size: Size of the square tiles to which the boxes are mapped.

    Returns:
        Tensor of shape ``(num_tiles_h, num_tiles_w)`` indicating which tiles contain objects.
        Tiles are indexed as ``mask[y, x]`` to match ViT's row-major patch order.
    """
    height, width = image_size
    num_tiles_h = int(height // tile_size)
    num_tiles_w = int(width // tile_size)
    mask = torch.zeros((num_tiles_h, num_tiles_w), dtype=torch.float32)
    for x1, y1, x2, y2 in boxes:
        # Boxes can have fractional coordinates after image resizing.
        full_x1 = max(0, math.ceil(x1 / tile_size))
        full_y1 = max(0, math.ceil(y1 / tile_size))
        full_x2 = min(num_tiles_w, int(x2 // tile_size))
        full_y2 = min(num_tiles_h, int(y2 // tile_size))
        if full_x1 < full_x2 and full_y1 < full_y2:
            mask[full_y1:full_y2, full_x1:full_x2] = 1
            continue

        tile_x1 = max(0, int(x1 // tile_size))
        tile_y1 = max(0, int(y1 // tile_size))
        tile_x2 = min(num_tiles_w, math.ceil(x2 / tile_size))
        tile_y2 = min(num_tiles_h, math.ceil(y2 / tile_size))
        mask[tile_y1:tile_y2, tile_x1:tile_x2] = 1
    return mask


class AmexDataset(Dataset):
    """Load AMEX screenshots and the corresponding element annotations.

    Args:
        screenshot_dir: Directory containing PNG screenshots.
        element_anno_dir: Directory containing JSON element annotations.  An
            annotation is paired with a screenshot when their file stems match.
        transform: Optional image transform. It receives the resized RGB CHW
            tensor returned by :func:`torchvision.io.read_image`.
        shrink_ratio: Factor applied to both image dimensions and annotation
            boxes before patch targets are calculated.

    Each item is ``(image, target)``.  With no transform, ``image`` is a
    float32 RGB tensor in ``[0, 1]`` with shape ``(C, H, W)``. ``target`` is
    a float32 tensor with shape ``(num_patches, 2)``: one row per DINO patch
    in row-major ``(y, x)`` order, with clickable and scrollable labels in the
    final dimension. Any bottom or right partial tile is omitted, matching the
    patch embedding's stride-``tile_size`` output.
    """

    def __init__(
        self,
        screenshot_dir: str | Path,
        element_anno_dir: str | Path,
        transform: Callable[[torch.Tensor], Any] | None = None,
        tile_size: int = 16,
        shrink_ratio: float = 0.25,
    ) -> None:
        self.screenshot_dir = Path(screenshot_dir)
        self.element_anno_dir = Path(element_anno_dir)
        self.transform = transform
        self.tile_size = tile_size
        self.shrink_ratio = shrink_ratio
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if not 0 < shrink_ratio <= 1:
            raise ValueError("shrink_ratio must be in the interval (0, 1]")
        if not self.screenshot_dir.is_dir():
            raise FileNotFoundError(
                f"Screenshot directory does not exist: {self.screenshot_dir}"
            )
        if not self.element_anno_dir.is_dir():
            raise FileNotFoundError(
                f"Annotation directory does not exist: {self.element_anno_dir}"
            )

        annotations = {path.stem: path for path in self.element_anno_dir.glob("*.json")}
        self.samples = [
            (image_path, annotations[image_path.stem])
            for image_path in sorted(self.screenshot_dir.rglob("*.png"))
            # if image_path.stem in annotations
        ]
        if not self.samples:
            raise RuntimeError("No PNG screenshots have a same-named JSON annotation.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, torch.Tensor]:
        image_path, annotation_path = self.samples[index]
        image = read_image(str(image_path), mode=ImageReadMode.RGB)
        _, height, width = image.shape
        resized_height = max(1, round(height * self.shrink_ratio))
        resized_width = max(1, round(width * self.shrink_ratio))
        image = resize(
            image,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        if self.transform:
            image = self.transform(image)
        with annotation_path.open("r", encoding="utf-8") as annotation_file:
            annotation = json.load(annotation_file)

        elements = list(annotation.get("clickable_elements", []))
        scrollable = annotation.get("scrollable_elements", [])
        scale_x = resized_width / width
        scale_y = resized_height / height

        def shrink_boxes(element_group: list[dict[str, Any]]) -> list[list[float]]:
            boxes = self._boxes(element_group, width, height)
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            return boxes.tolist()

        # DINO's patch embedding uses a stride equal to the patch size, so it
        # emits floor(H / tile_size) * floor(W / tile_size) patch tokens.
        # Keep the two spatial axes until both classes have been filled, then
        # flatten in row-major (y, x) order to match ViT's patch-token order.
        mask = torch.zeros(
            (
                resized_height // self.tile_size,
                resized_width // self.tile_size,
                2,
            ),
            dtype=torch.float32,
        )

        mask[:, :, 0] = bboxes_to_mask(
            shrink_boxes(elements),
            (resized_height, resized_width),
            tile_size=self.tile_size,
        )
        mask[:, :, 1] = bboxes_to_mask(
            shrink_boxes(scrollable),
            (resized_height, resized_width),
            tile_size=self.tile_size,
        )

        return image, mask.reshape(-1, 2)

    @staticmethod
    def _boxes(elements: list[dict[str, Any]], width: int, height: int) -> torch.Tensor:
        boxes: list[list[float]] = []
        for element in elements:
            bbox = element.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                raise ValueError(f"Invalid bbox in annotation: {bbox!r}")
            x1, y1, x2, y2 = (float(value) for value in bbox)
            # Keep annotations valid if a box extends one or two pixels outside
            # an image due to rounding in the source data.
            boxes.append(
                [
                    max(0.0, min(x1, float(width))),
                    max(0.0, min(y1, float(height))),
                    max(0.0, min(x2, float(width))),
                    max(0.0, min(y2, float(height))),
                ]
            )
        return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)

    @staticmethod
    def collate_fn(
        batch: list[tuple[Any, torch.Tensor]],
    ) -> tuple[list[Any], list[torch.Tensor]]:
        """Preserve variable-size screenshots and their patch targets as lists.

        Screenshots have different native heights, which also means their DINO
        patch-target sequences have different lengths.  They therefore cannot
        be stacked into dense tensors here.
        """
        images, targets = zip(*batch)
        return list(images), list(targets)


class ProccessedDataset(Dataset):
    def __init__(self, feature_dir: str | Path, target_dir: str | Path):
        self.feature_dir = Path(feature_dir)
        self.target_dir = Path(target_dir)
        feature_stems = {
            path.stem.split("_")[-1] for path in self.feature_dir.rglob("*.pt")
        }
        target_stems = {
            path.stem.split("_")[-1] for path in self.target_dir.rglob("*.pt")
        }
        self.samples = sorted(list(feature_stems.intersection(target_stems)))
        if not self.samples:
            raise RuntimeError(
                "No processed features have a same-named processed target."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        feature_path = self.feature_dir / f"feature_{self.samples[index]}.pt"
        target_path = self.target_dir / f"target_{self.samples[index]}.pt"
        feature = torch.load(feature_path, map_location=torch.device("cpu")).squeeze(0)
        target = torch.load(target_path, map_location=torch.device("cpu")).squeeze(0)
        return feature, target


def proccess_amex_to_tensor():
    dataset = AmexDataset("D:\\screenshot", "element_anno")
    print(f"Loaded {len(dataset)} samples from screenshot and element_anno")

    from dinov3 import DINOv3FeatureExtractor
    import tqdm

    dino = DINOv3FeatureExtractor()
    from torch import save

    proccessed_samples = [
        image_path.stem for image_path in sorted(Path("feature").rglob("*.pt"))
    ]
    proccessed_samples = {int(x.split("_")[-1]) for x in proccessed_samples}

    proccessed_targets = [
        target_path.stem for target_path in sorted(Path("target").rglob("*.pt"))
    ]
    proccessed_targets = {int(x.split("_")[-1]) for x in proccessed_targets}

    proccessed = proccessed_samples.intersection(proccessed_targets)

    bad_samples = []
    for i in tqdm.tqdm(range(len(dataset)), desc="Processing samples"):
        try:
            if i in proccessed:
                continue
            image, target = dataset[i]
            if i not in proccessed_samples:
                feature = dino(image)
                save(feature, f"feature/feature_{i}.pt")
            if i not in proccessed_targets:
                save(target, f"target/target_{i}.pt")
        except Exception as e:
            bad_samples.append(dataset.samples[i][0].name)
    with open("bad_samples.txt", "w", encoding="utf-8") as f:
        for sample in bad_samples:
            f.write(sample + "\n")

    print(f"Found {len(bad_samples)} bad samples. Check 'bad_samples.txt' for details.")


if __name__ == "__main__":
    """
    import argparse

    parser = argparse.ArgumentParser(description="Test AMEX dataset")
    parser.add_argument("screenshot_dir", help="Directory containing PNG screenshots")
    parser.add_argument("element_anno_dir", help="Directory containing JSON element annotations")
    args = parser.parse_args()
    """

    dataset = ProccessedDataset("feature", "target")
    print(f"Loaded {len(dataset)} samples from processed features and targets")
