import torch
import torch.nn as nn

from .network import Encoder, DynamicTanh, SwiGLU


class ElementFinder(nn.Module):
    num_special_tokens = 5

    def __init__(self) -> None:
        super().__init__()
        dim = 384  # DINOv3 ViT-S/16 embedding dimension
        self.model = Encoder(dim, num_layers=2)
        self.head = nn.Sequential(
            DynamicTanh(dim),
            SwiGLU(dim, dim),
            nn.Linear(dim, 2),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.model(images)
        logits = self.head(x)
        logits = logits[:, self.num_special_tokens :]
        return logits

    def process_image(self, images: torch.Tensor):
        """
        ``images`` is channel-first (``C, H, W``) or batched channel-first
        (``B, C, H, W``). The result has shape ``(B, patches, 2)``; for an
        unbatched image, the batch dimension is removed to match
        :class:`AmexDataset`'s ``(patches, 2)`` target.
        """
        """Return one two-class logit vector per image patch."""
        """
        is_unbatched = images.ndim == 3
        if images.ndim not in (3, 4):
            raise ValueError(
                "images must have shape (C, H, W) or (B, C, H, W), "
                f"got {tuple(images.shape)}"
            )
        channel_dim = 0 if is_unbatched else 1
        if images.shape[channel_dim] != 3:
            raise ValueError(
                "images must be RGB and channel-first: expected 3 channels at "
                f"dimension {channel_dim}, got shape {tuple(images.shape)}"
            )
        """
        pass
        # return logits.squeeze(0) if is_unbatched else logits
