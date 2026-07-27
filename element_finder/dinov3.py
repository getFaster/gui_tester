import torch
from transformers import AutoImageProcessor, AutoModel


class DINOv3FeatureExtractor:
    def __init__(self, model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m"):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            device_map="auto",
        )
        self.patch_size = 16
        # The sequence starts with CLS and may include DINO register tokens.
        # Keep this model-specific detail out of the training loop.
        self.num_special_tokens = 1 + int(
            getattr(self.model.config, "num_register_tokens", 0)
        )

    def __call__(self, img) -> torch.Tensor:
        inputs = self.processor(images=img, do_resize=False, return_tensors="pt").to(
            self.model.device
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state
