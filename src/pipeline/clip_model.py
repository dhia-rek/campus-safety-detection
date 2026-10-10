"""Load CLIP ViT-B/32 with OpenAI's weights through open_clip.

The original code used the `openai-clip` package, which is unmaintained and
breaks on current setuptools (it imports `pkg_resources`). open_clip ships the
same OpenAI checkpoint. The `-quickgelu` variant matters: OpenAI's weights were
trained with QuickGELU, and open_clip's plain "ViT-B-32" config uses GELU.
Checked: text embeddings match openai-clip to cosine 1.0000 on prompts/*.txt.
"""

from __future__ import annotations

MODEL_NAME = "ViT-B-32-quickgelu"
PRETRAINED = "openai"


def load_clip(device: str = "cpu"):
    """Return (model, preprocess, tokenizer), model in eval mode on *device*."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED, device=device)
    model.eval()
    return model, preprocess, open_clip.get_tokenizer(MODEL_NAME)
