from __future__ import annotations

import math

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from configs.config import Config
from data.prompted_dataset import make_prompted_data_loaders
from losses.contrastive import symmetric_infonce_loss
from models import ModelHandler
from models.medstyle_3dg import (
    MedStyle3DG,
    build_biomed_text_encoder_from_config,
)


def _freeze_assertions(model: MedStyle3DG) -> None:
    frozen_prefixes = ("vision_encoder", "text_encoder")
    for name, param in model.named_parameters():
        if name.startswith(frozen_prefixes) and param.requires_grad:
            raise AssertionError(f"Encoder parameter should be frozen: {name}")
    trainable = model.trainable_parameter_names()
    allowed = ("image_projection", "text_projection", "log_temperature")
    unexpected = [name for name in trainable if not name.startswith(allowed)]
    if unexpected:
        raise AssertionError(f"Unexpected trainable parameters: {unexpected}")


class Phase2AlignmentTrainer:
    def __init__(
        self,
        config: Config | None = None,
        device_id: int = 0,
        vision_encoder: torch.nn.Module | None = None,
        text_encoder: torch.nn.Module | None = None,
    ):
        self.config = config or Config()
        self.config.batch_size = self.config.medstyle_alignment_batch_size
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.train_loader, self.val_loader, self.test_loader = make_prompted_data_loaders(self.config)
        self.vision_encoder = vision_encoder or self._build_vision_encoder()
        self.text_encoder = text_encoder or self._build_text_encoder()
        self.vision_dim = getattr(self.vision_encoder, "feature_dim", None)
        if self.vision_dim is None and hasattr(self.vision_encoder, "backbone"):
            self.vision_dim = getattr(self.vision_encoder.backbone, "feature_dim", None)
        if self.vision_dim is None:
            raise ValueError("vision_encoder must expose feature_dim")
        self.text_dim = int(getattr(self.text_encoder, "output_dim"))
        self.model = MedStyle3DG(
            vision_encoder=self.vision_encoder,
            text_encoder=self.text_encoder,
            vision_dim=self.vision_dim,
            text_dim=self.text_dim,
            latent_dim=self.config.medstyle_latent_dim,
            temperature_init=self.config.medstyle_temperature_init,
            freeze_encoders=self.config.medstyle_freeze_encoders,
        ).to(self.device)
        _freeze_assertions(self.model)
        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.medstyle_alignment_lr,
            weight_decay=self.config.medstyle_alignment_weight_decay,
        )
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self._lr_factor)

    def _lr_factor(self, step: int) -> float:
        warmup_steps = max(1, self.config.medstyle_alignment_warmup_epochs * max(1, len(self.train_loader)))
        total_steps = max(warmup_steps + 1, self.config.medstyle_alignment_epochs * max(1, len(self.train_loader)))
        if step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item()

    def _build_vision_encoder(self):
        model = ModelHandler(self.config).initialize_model(self.config.model_to_be_used)
        if self.config.medstyle_phase1_checkpoint:
            checkpoint = torch.load(self.config.medstyle_phase1_checkpoint, map_location="cpu")
            state = checkpoint.get("vision_encoder_state_dict") or checkpoint.get("model_state_dict") or checkpoint
            model.load_state_dict(state, strict=False)
        return model

    def _build_text_encoder(self):
        return build_biomed_text_encoder_from_config(self.config)

    def train_step(self, batch):
        images, _labels, _domains, prompts, _metadata = batch
        images = images.float().to(self.device)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        image_emb, text_emb = self.model(images, prompts)
        loss, logits = symmetric_infonce_loss(
            image_emb,
            text_emb,
            self.model.log_temperature,
            min_temperature=self.config.medstyle_temperature_min,
        )
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            self.model.log_temperature.clamp_(min=math.log(float(self.config.medstyle_temperature_min)))
        self.scheduler.step()
        return float(loss.detach().cpu()), logits.detach()


__all__ = ["Phase2AlignmentTrainer"]
