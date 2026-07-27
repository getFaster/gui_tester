## File Structure

- `train.py` trains and validates the element classifier and writes TensorBoard
  logs and checkpoints.
- `amex_dataset.py` loads AMEX screenshots and annotations, builds patch masks,
  and loads precomputed features.
- `dinov3.py` wraps DINOv3 feature extraction.
- `element_finder.py` defines the element-classification model.
- `network.py` contains reusable neural-network layers.
- `pipeline.py` and `predict.py` provide feature-generation and inference
  workflows.


## Data and Model Contracts

- DINOv3 uses 16 by 16 patches. Build masks as `[y, x]` and flatten them in  row-major order so targets align with patch tokens.
- Remove the five DINOv3 special tokens before computing patch-target loss.
- Processed-feature training groups samples by exact token count. Each dense
  sub-batch contains equal-length sequences, while one logical batch still
  produces one optimizer step.
- Use `torchvision.io.read_image` and established PyTorch/torchvision APIs
  instead of adding bespoke image-conversion helpers.
