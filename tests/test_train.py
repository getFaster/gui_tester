import unittest

import torch
import torch.nn.functional as F

from element_finder.train import batch_loss


class RecordingModel(torch.nn.Module):
    num_special_tokens = 5

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.75))
        self.forward_shapes: list[tuple[int, ...]] = []

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self.forward_shapes.append(tuple(features.shape))
        return features[:, self.num_special_tokens :, :2] * self.scale


def make_sample(token_count: int, offset: float) -> tuple[torch.Tensor, torch.Tensor]:
    feature = torch.arange(token_count * 3, dtype=torch.float32).reshape(
        token_count, 3
    )
    feature = feature / 10 + offset
    target = (
        torch.arange((token_count - 5) * 2).reshape(token_count - 5, 2) % 2
    ).float()
    return feature, target


class BatchLossTests(unittest.TestCase):
    def test_groups_by_token_count_and_matches_reference(self) -> None:
        samples = [
            make_sample(9, 0.0),
            make_sample(11, 0.1),
            make_sample(9, 0.2),
            make_sample(11, 0.3),
            make_sample(9, 0.4),
        ]
        features = [feature for feature, _ in samples]
        targets = [target for _, target in samples]
        model = RecordingModel()
        loss_fn = torch.nn.BCEWithLogitsLoss()

        expected_losses = [
            F.binary_cross_entropy_with_logits(
                feature[model.num_special_tokens :, :2] * model.scale,
                target,
            )
            for feature, target in samples
        ]
        expected_loss = torch.stack(expected_losses).mean()
        expected_loss.backward()
        expected_gradient = model.scale.grad.detach().clone()
        model.scale.grad = None

        actual_loss = batch_loss(
            model,
            features,
            targets,
            loss_fn,
            torch.device("cpu"),
            sub_batch_size=2,
            backward=True,
        )

        self.assertEqual(
            model.forward_shapes,
            [(2, 9, 3), (1, 9, 3), (2, 11, 3)],
        )
        self.assertAlmostEqual(actual_loss, expected_loss.item())
        torch.testing.assert_close(model.scale.grad, expected_gradient)

    def test_rejects_mismatched_sample_counts(self) -> None:
        feature, target = make_sample(9, 0.0)

        with self.assertRaisesRegex(ValueError, "same number of samples"):
            batch_loss(
                RecordingModel(),
                [feature],
                [target, target],
                torch.nn.BCEWithLogitsLoss(),
                torch.device("cpu"),
                sub_batch_size=2,
                backward=False,
            )

    def test_rejects_incorrect_target_length(self) -> None:
        feature, target = make_sample(9, 0.0)

        with self.assertRaisesRegex(ValueError, "one row per patch token"):
            batch_loss(
                RecordingModel(),
                [feature],
                [target[:-1]],
                torch.nn.BCEWithLogitsLoss(),
                torch.device("cpu"),
                sub_batch_size=2,
                backward=False,
            )


if __name__ == "__main__":
    unittest.main()
