import tempfile
import unittest
from pathlib import Path

import torch
from torchvision.io import write_png

from pipeline import (
    eager_softcap_attention,
    load_images_from_directory,
    render_probability_overlay,
    resize_image,
)


class PipelineTest(unittest.TestCase):
    def test_load_images_recursively_and_ignore_other_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            nested = root / "nested"
            nested.mkdir()
            write_png(torch.zeros((3, 2, 2), dtype=torch.uint8), str(root / "b.png"))
            write_png(torch.zeros((3, 2, 2), dtype=torch.uint8), str(nested / "a.png"))
            (root / "notes.txt").write_text("not an image", encoding="utf-8")

            paths = load_images_from_directory(root)

            self.assertEqual(paths, sorted([root / "b.png", nested / "a.png"]))

    def test_resize_image_uses_height_width_order(self) -> None:
        image = torch.zeros((3, 20, 40), dtype=torch.uint8)

        resized = resize_image(image, 0.25)

        self.assertEqual(resized.shape, (3, 5, 10))

    def test_overlay_maps_probability_grid_in_row_major_order(self) -> None:
        image = torch.zeros((3, 5, 5), dtype=torch.uint8)
        probabilities = torch.tensor([[1.0, 0.0], [0.0, 0.5]])

        overlay = render_probability_overlay(
            image, probabilities, tile_size=2, opacity=1.0
        )

        self.assertTrue(torch.equal(overlay[:, 0, 0], torch.tensor([0.0, 0.0, 1.0])))
        self.assertTrue(torch.equal(overlay[:, 0, 2], torch.zeros(3)))
        self.assertTrue(
            torch.equal(overlay[:, 2, 2], torch.tensor([0.0, 0.0, 0.5]))
        )
        self.assertTrue(torch.equal(overlay[:, 4, 4], torch.zeros(3)))

    def test_eager_attention_uses_uniform_weights_for_zero_queries(self) -> None:
        query = torch.zeros((1, 1, 2, 2))
        key = torch.tensor([[[[2.0, 1.0], [1.0, 2.0]]]])
        value = torch.tensor([[[[1.0, 3.0], [5.0, 7.0]]]])

        output = eager_softcap_attention(query, key, value)

        expected = torch.tensor([[[[3.0, 5.0], [3.0, 5.0]]]])
        self.assertTrue(torch.equal(output, expected))


if __name__ == "__main__":
    unittest.main()
