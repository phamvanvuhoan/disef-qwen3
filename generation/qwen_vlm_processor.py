"""Qwen VLM helpers used by the DISEF generation pipeline.

These replace the original LLaVA (captioning) + CLIP (scoring) combination:

- ``QwenCaptioner`` wraps a *generative* Qwen VL model (e.g. Qwen2.5-VL) and
  writes a textual description for a real image (TODO.md step 2).
- ``QwenEmbedder`` wraps ``Qwen/Qwen3-VL-Embedding-2B`` and produces image/text
  embeddings used to score synthetic images by cosine similarity to their class
  (TODO.md step 3).
"""

from typing import List, Union

import torch
from PIL import Image


class QwenCaptioner:
    """Generative Qwen VL captioner.

    Note: this needs a *generative* checkpoint (e.g. ``Qwen/Qwen2.5-VL-7B-Instruct``).
    ``Qwen/Qwen3-VL-Embedding-2B`` is embedding-only and cannot caption.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.model_name_or_path = model_name_or_path
        self.device = device

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.tokenizer = self.processor.tokenizer

    def caption_image(
        self,
        image: Union[str, Image.Image],
        class_name: str,
        max_new_tokens: int = 256,
    ) -> str:
        """Return a detailed caption for ``image`` focused on ``class_name``."""
        query = (
            "Provide a detailed caption for the image focusing on the object, "
            f"knowing it's a {class_name}."
        )
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": query},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(
            self.device
        )

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        caption = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return caption


class QwenEmbedder:
    """Qwen3-VL-Embedding wrapper for image/text embedding and scoring.

    Requires ``pip install "sentence-transformers[image]"``.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-VL-Embedding-2B",
        device: str = "cuda",
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name_or_path = model_name_or_path
        self.model = SentenceTransformer(
            model_name_or_path, device=device, trust_remote_code=True
        )
        self.model.eval()

    @torch.no_grad()
    def embed_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode texts to a normalized embedding tensor of shape ``(N, D)``."""
        return self.model.encode(
            texts, convert_to_tensor=True, normalize_embeddings=True
        )

    @torch.no_grad()
    def embed_images(self, images: List[Image.Image]) -> torch.Tensor:
        """Encode images to a normalized embedding tensor of shape ``(N, D)``."""
        return self.model.encode(
            images, convert_to_tensor=True, normalize_embeddings=True
        )
