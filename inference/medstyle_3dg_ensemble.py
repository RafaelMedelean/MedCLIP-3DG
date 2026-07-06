from __future__ import annotations

import torch
import torch.nn.functional as F

from data.prompted_dataset import NEGATIVE_PROMPT, POSITIVE_SIMPLE


def medstyle_similarity_logits(medstyle_model, image_emb: torch.Tensor, text_emb: torch.Tensor):
    logits = image_emb @ text_emb.t()
    if hasattr(medstyle_model, "log_temperature"):
        logits = logits / medstyle_model.log_temperature.exp().clamp_min(1e-8)
    return logits


def canonical_prompt_probability(medstyle_model, image_emb: torch.Tensor):
    prompts = [NEGATIVE_PROMPT, POSITIVE_SIMPLE]
    text_emb = medstyle_model.encode_text(prompts)
    logits = medstyle_similarity_logits(medstyle_model, image_emb, text_emb)
    return F.softmax(logits, dim=-1)[:, 1]


class MedStyle3DGEnsemble:
    def __init__(self, medstyle_model, style_model, style_bank, weights=(1 / 3, 1 / 3, 1 / 3), threshold: float = 0.5):
        self.medstyle = medstyle_model
        self.style_model = style_model
        self.style_bank = style_bank
        self.weights = tuple(float(w) for w in weights)
        self.threshold = float(threshold)

    @torch.no_grad()
    def branch_probabilities(self, images: torch.Tensor, k_styles: int = 20):
        self.medstyle.eval()
        self.style_model.eval()
        device = next(self.medstyle.parameters()).device
        images = images.float().to(device)
        image_emb = self.medstyle.encode_image(images)

        p1 = canonical_prompt_probability(self.medstyle, image_emb)

        filtered = self.style_model.style_se(image_emb)
        logits2 = self.style_model.classifier(filtered)
        p2 = F.softmax(logits2, dim=-1)[:, 1]

        style_vectors, _weights = self.style_bank.sample(k_styles)
        styled_positive = self.medstyle.encode_text_with_style_embeddings(
            self.style_bank.class_style_prompts([1], k_styles),
            style_vectors,
        )
        styled_negative = self.medstyle.encode_text_with_style_embeddings(
            self.style_bank.class_style_prompts([0], k_styles),
            style_vectors,
        )
        pos_logits = medstyle_similarity_logits(self.medstyle, image_emb, styled_positive)
        neg_logits = medstyle_similarity_logits(self.medstyle, image_emb, styled_negative)
        stacked = torch.stack((neg_logits, pos_logits), dim=-1)
        p3 = F.softmax(stacked, dim=-1)[:, :, 1].mean(dim=1)

        return p1, p2, p3

    @torch.no_grad()
    def predict_proba(self, images: torch.Tensor, k_styles: int = 20):
        p1, p2, p3 = self.branch_probabilities(images, k_styles=k_styles)
        w1, w2, w3 = self.weights
        return w1 * p1 + w2 * p2 + w3 * p3

    @torch.no_grad()
    def predict(self, images: torch.Tensor, k_styles: int = 20):
        probabilities = self.predict_proba(images, k_styles=k_styles)
        return (probabilities >= self.threshold).long(), probabilities


__all__ = ["MedStyle3DGEnsemble", "canonical_prompt_probability", "medstyle_similarity_logits"]
