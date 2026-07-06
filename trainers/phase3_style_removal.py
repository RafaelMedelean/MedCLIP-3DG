from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import SGD

from configs.config import Config
from data.prompted_dataset import make_prompted_data_loaders
from style.arcface import ArcFaceClassifier, arcface_loss
from style.domain_uncertainty import domain_uncertainty_loss, style_similarity_logits
from style.dpstyler import SourceStyleBank
from style.style_se import StyleSE


class Phase3StyleRemovalModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512,
        num_classes: int = 2,
        num_styles: int = 20,
        reduction: int = 16,
        arcface_scale: float = 5.0,
        arcface_margin: float = 0.5,
    ):
        super().__init__()
        self.style_se = StyleSE(embedding_dim=embedding_dim, reduction=reduction)
        self.classifier = ArcFaceClassifier(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            scale=arcface_scale,
            margin=arcface_margin,
        )
        self.num_styles = num_styles

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None):
        filtered = self.style_se(embeddings)
        return self.classifier(filtered, labels), filtered


class Phase3StyleRemovalTrainer:
    def __init__(self, medstyle_model, config: Config | None = None, device_id: int = 0, source_domain_names=None):
        self.config = config or Config()
        self.config.batch_size = self.config.style_batch_size
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.medstyle = medstyle_model.to(self.device)
        self.medstyle.eval()
        for param in self.medstyle.parameters():
            param.requires_grad = False
        self.train_loader, self.val_loader, self.test_loader = make_prompted_data_loaders(self.config)
        domain_names = list(source_domain_names or self.config.source_domain_names or self._infer_domain_names())
        self.style_bank = SourceStyleBank.from_medstyle(
            self.medstyle,
            domain_names,
            beta=self.config.style_beta,
        )
        self.model = Phase3StyleRemovalModel(
            embedding_dim=self.config.medstyle_latent_dim,
            num_classes=2,
            num_styles=self.config.style_k,
            reduction=self.config.style_se_reduction,
            arcface_scale=self.config.style_arcface_scale,
            arcface_margin=self.config.style_arcface_margin,
        ).to(self.device)
        self.optimizer = SGD(
            self.model.parameters(),
            lr=self.config.style_lr,
            momentum=self.config.style_momentum,
        )

    def _infer_domain_names(self) -> list[str]:
        dataset = getattr(self.train_loader, "dataset", None)
        frame = getattr(dataset, "frame", None)
        if frame is not None and self.config.domain_column in frame.columns:
            return sorted(str(v) for v in frame[self.config.domain_column].dropna().unique())
        return ["source"]

    def _styled_text_embeddings(self, prompts: list[str], labels: torch.Tensor):
        with torch.no_grad():
            style_vectors, _ = self.style_bank.sample(self.config.style_k)
            labels_cpu = labels.detach().cpu().long().tolist()
            style_prompts = self.style_bank.class_style_prompts(labels_cpu, self.config.style_k)
            repeated_style_vectors = style_vectors.unsqueeze(0).expand(labels.shape[0], -1, -1).reshape(
                labels.shape[0] * self.config.style_k,
                -1,
            )
            styled = self.medstyle.encode_text_with_style_embeddings(style_prompts, repeated_style_vectors)
            style_only_prompts = self.style_bank.style_only_prompts(self.config.style_k)
            style_features = self.medstyle.encode_text_with_style_embeddings(style_only_prompts, style_vectors)
        labels_rep = labels.long().unsqueeze(1).expand(-1, self.config.style_k).reshape(-1)
        return styled.reshape(-1, styled.shape[-1]), labels_rep, style_features

    def train_step(self, batch):
        _images, labels, _domains, prompts, _metadata = batch
        labels = labels.long().to(self.device)
        embeddings, labels_rep, style_features = self._styled_text_embeddings(prompts, labels)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits, filtered = self.model(embeddings, labels_rep)
        cls_loss = arcface_loss(self.model.classifier, filtered, labels_rep)
        style_logits = style_similarity_logits(
            filtered,
            style_features,
            scale=self.config.style_uncertainty_logit_scale,
        )
        uncertainty = domain_uncertainty_loss(style_logits)
        loss = cls_loss + self.config.style_uncertainty_weight * uncertainty
        loss.backward()
        self.optimizer.step()
        return {
            "loss": float(loss.detach().cpu()),
            "arcface_loss": float(cls_loss.detach().cpu()),
            "uncertainty_loss": float(uncertainty.detach().cpu()),
            "uncertainty_entropy": float((-uncertainty).detach().cpu()),
            "style_logit_scale": float(self.config.style_uncertainty_logit_scale),
        }

    @torch.no_grad()
    def image_branch_logits(self, images: torch.Tensor):
        images = images.float().to(self.device)
        image_emb = self.medstyle.encode_image(images)
        filtered = self.model.style_se(image_emb)
        return self.model.classifier(filtered)


__all__ = ["Phase3StyleRemovalModel", "Phase3StyleRemovalTrainer"]
