# VLA Playground

This workspace is reserved for local ViT, CLIP, and VLA-related experiments.

Initial environment:

```text
conda env: cv-train
Python: 3.11
PyTorch: CUDA-enabled
GPU: RTX 4060 Laptop
```

Installed exploration packages:

```text
timm
open_clip_torch
transformers
accelerate
einops
```

Suggested first steps:

1. CLIP zero-shot image-text similarity on a small image folder.
2. ViT/CLIP image embeddings and nearest-neighbor retrieval.
3. t-SNE or UMAP visualization of image embeddings.
4. Small vision-language grounding experiments before moving to full VLA models.

No model weights or datasets are committed here by default.
