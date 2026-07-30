import unittest

import torch

from element_finder.predict import best_f1_operating_point, collect_predictions


class RecordingModel(torch.nn.Module):
    num_special_tokens = 5

    def __init__(self) -> None:
        super().__init__()
        self.forward_shapes: list[tuple[int, ...]] = []

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self.forward_shapes.append(tuple(features.shape))
        return features[:, self.num_special_tokens :, :2]


class PredictionTests(unittest.TestCase):
    def test_collect_predictions_groups_ragged_samples(self) -> None:
        model = RecordingModel()
        features = [
            torch.zeros(7, 3),
            torch.ones(8, 3),
            torch.full((7, 3), 2.0),
        ]
        targets = [
            torch.zeros(2, 2),
            torch.ones(3, 2),
            torch.ones(2, 2),
        ]

        probabilities, actual_targets = collect_predictions(
            model,
            [(features, targets)],
            torch.device("cpu"),
            sub_batch_size=2,
        )

        self.assertEqual(model.forward_shapes, [(2, 7, 3), (1, 8, 3)])
        self.assertEqual(probabilities.shape, (7, 2))
        self.assertEqual(actual_targets.shape, (7, 2))
        torch.testing.assert_close(actual_targets.sum(), torch.tensor(10.0))

    def test_best_f1_uses_complete_tied_threshold_group(self) -> None:
        probabilities = torch.tensor([0.9, 0.8, 0.8, 0.1])
        targets = torch.tensor([1, 0, 1, 0])

        threshold, precision, recall, f1 = best_f1_operating_point(
            probabilities, targets
        )

        self.assertAlmostEqual(threshold, 0.8)
        self.assertAlmostEqual(precision, 2 / 3)
        self.assertAlmostEqual(recall, 1.0)
        self.assertAlmostEqual(f1, 0.8)

    def test_best_f1_rejects_different_element_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "same number of elements"):
            best_f1_operating_point(torch.tensor([0.5]), torch.tensor([0, 1]))


if __name__ == "__main__":
    unittest.main()
