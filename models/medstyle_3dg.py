from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


STYLE_TOKEN = "[STYLE]"


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad = False


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 512):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class BiomedTextEncoder(nn.Module):
    def __init__(self, model_name_or_path: str, trust_remote_code: bool = False, style_token: str = STYLE_TOKEN):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:
            raise ImportError(
                "transformers is required for BiomedVLP-CXR-BERT. Install transformers and configure "
                "GENERALIZATION_TEXT_MODEL_PATH or GENERALIZATION_TEXT_MODEL_NAME."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        self.style_token = style_token
        added = self.tokenizer.add_special_tokens({"additional_special_tokens": [style_token]})
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        if added:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.output_dim = int(self.model.config.hidden_size)
        freeze_module(self.model)

    def _pool_outputs(self, outputs):
        if getattr(outputs, "pooler_output", None) is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0]

    def style_embeddings(self, prompts: list[str]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        batch = self.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            embedded = self.model.get_input_embeddings()(batch["input_ids"])
            mask = batch["attention_mask"].unsqueeze(-1).to(embedded.dtype)
            return (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

    def forward_with_style_embeddings(
        self,
        prompts: list[str],
        style_embeddings: torch.Tensor,
        style_token: str = STYLE_TOKEN,
    ) -> torch.Tensor:
        device = next(self.model.parameters()).device
        batch = self.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
        input_ids = batch.pop("input_ids")
        input_embeds = self.model.get_input_embeddings()(input_ids).clone()
        style_id = self.tokenizer.convert_tokens_to_ids(style_token)
        style_mask = input_ids == style_id
        if not torch.all(style_mask.any(dim=1)):
            raise ValueError(f"Every prompt must contain the style token {style_token!r}")
        style_embeddings = style_embeddings.to(device=input_embeds.device, dtype=input_embeds.dtype)
        if style_embeddings.ndim == 1:
            style_embeddings = style_embeddings.unsqueeze(0).expand(input_ids.shape[0], -1)
        if style_embeddings.shape[0] != input_ids.shape[0]:
            raise ValueError("style_embeddings must have one row per prompt")
        input_embeds[style_mask] = style_embeddings.repeat_interleave(style_mask.sum(dim=1), dim=0)
        with torch.no_grad():
            if hasattr(self.model, "bert"):
                outputs = self.model.bert(
                    inputs_embeds=input_embeds,
                    attention_mask=batch.get("attention_mask"),
                    token_type_ids=batch.get("token_type_ids"),
                    return_dict=True,
                )
            else:
                outputs = self.model(inputs_embeds=input_embeds, **batch)
            return self._pool_outputs(outputs)

    def forward(self, prompts: list[str]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        batch = self.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model(**batch)
            return self._pool_outputs(outputs)


class MedStyle3DG(nn.Module):
    def __init__(
        self,
        vision_encoder: nn.Module,
        text_encoder: nn.Module,
        vision_dim: int,
        text_dim: int,
        latent_dim: int = 512,
        temperature_init: float = 0.07,
        freeze_encoders: bool = True,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.freeze_encoders = bool(freeze_encoders)
        if self.freeze_encoders:
            freeze_module(self.vision_encoder)
            freeze_module(self.text_encoder)
        self.image_projection = ProjectionHead(vision_dim, latent_dim)
        self.text_projection = ProjectionHead(text_dim, latent_dim)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(float(temperature_init))))
        self.latent_dim = latent_dim

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoders:
            self.vision_encoder.eval()
            self.text_encoder.eval()
        return self

    def encode_image(self, images: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        with torch.no_grad():
            if hasattr(self.vision_encoder, "forward_features"):
                features = self.vision_encoder.forward_features(images)
            else:
                output = self.vision_encoder(images, return_features=True)
                features = output[1] if isinstance(output, tuple) else output
        projected = self.image_projection(features)
        return F.normalize(projected, dim=-1) if normalize else projected

    def encode_text(self, prompts_or_tokens: Any, normalize: bool = True) -> torch.Tensor:
        with torch.no_grad():
            features = self.text_encoder(prompts_or_tokens)
        projected = self.text_projection(features)
        return F.normalize(projected, dim=-1) if normalize else projected

    def encode_text_with_style_embeddings(
        self,
        prompts: list[str],
        style_embeddings: torch.Tensor,
        normalize: bool = True,
        style_token: str = STYLE_TOKEN,
    ) -> torch.Tensor:
        if not hasattr(self.text_encoder, "forward_with_style_embeddings"):
            raise TypeError("text_encoder must implement forward_with_style_embeddings")
        with torch.no_grad():
            features = self.text_encoder.forward_with_style_embeddings(prompts, style_embeddings, style_token)
        projected = self.text_projection(features)
        return F.normalize(projected, dim=-1) if normalize else projected

    def text_style_embeddings(self, prompts: list[str]) -> torch.Tensor:
        if not hasattr(self.text_encoder, "style_embeddings"):
            raise TypeError("text_encoder must implement style_embeddings")
        return self.text_encoder.style_embeddings(prompts)

    def forward(self, images: torch.Tensor, prompts_or_tokens: Any):
        return self.encode_image(images), self.encode_text(prompts_or_tokens)

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, param in self.named_parameters() if param.requires_grad]


def build_biomed_text_encoder_from_config(config) -> BiomedTextEncoder:
    path = config.medstyle_text_model_path or config.medstyle_text_model_name
    return BiomedTextEncoder(path, trust_remote_code=config.medstyle_trust_remote_code)


__all__ = [
    "BiomedTextEncoder",
    "MedStyle3DG",
    "ProjectionHead",
    "STYLE_TOKEN",
    "build_biomed_text_encoder_from_config",
    "freeze_module",
]
