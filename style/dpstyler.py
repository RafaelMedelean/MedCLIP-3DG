from __future__ import annotations

import torch
import torch.nn as nn

from data.prompted_dataset import NEGATIVE_STYLE_TEMPLATE, POSITIVE_STYLE_TEMPLATE
from models.medstyle_3dg import STYLE_TOKEN


class SourceStyleBank(nn.Module):
    def __init__(self, domain_names, embeddings: torch.Tensor, beta: float = 0.1, style_token: str = STYLE_TOKEN):
        super().__init__()
        self.domain_names = list(domain_names)
        self.beta = beta
        self.style_token = style_token
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be [num_domains, embedding_dim]")
        self.register_buffer("domain_embeddings", embeddings.detach().float())

    @classmethod
    @torch.no_grad()
    def from_medstyle(cls, medstyle_model, domain_names, beta: float = 0.1, template: str = "{domain} CT acquisition style"):
        prompts = [template.format(domain=name) for name in domain_names]
        device = next(medstyle_model.parameters()).device
        if hasattr(medstyle_model, "text_style_embeddings"):
            embeddings = medstyle_model.text_style_embeddings(prompts).to(device)
        else:
            embeddings = medstyle_model.encode_text(prompts).to(device)
        return cls(domain_names, embeddings, beta=beta).to(device)

    def sample(self, n_styles: int = 20):
        device = self.domain_embeddings.device
        if self.domain_embeddings.shape[0] == 1:
            weights = torch.ones((n_styles, 1), device=device)
        else:
            concentration = torch.full((self.domain_embeddings.shape[0],), self.beta, device=device)
            weights = torch.distributions.Dirichlet(concentration).sample((n_styles,))
        mixed = weights @ self.domain_embeddings
        return mixed, weights

    def class_style_prompts(self, labels, n_styles: int = 20) -> list[str]:
        prompts = []
        for label in labels:
            template = POSITIVE_STYLE_TEMPLATE if int(label) == 1 else NEGATIVE_STYLE_TEMPLATE
            for _ in range(n_styles):
                prompts.append(template.format(style=self.style_token))
        return prompts

    def style_only_prompts(self, n_styles: int = 20) -> list[str]:
        return [f"{self.style_token} CT acquisition style." for _ in range(n_styles)]

    def style_prompt_texts(self, weights: torch.Tensor) -> list[str]:
        texts = []
        weights = weights.detach().cpu()
        for row in weights:
            parts = []
            for name, weight in zip(self.domain_names, row.tolist()):
                if weight >= 0.05:
                    parts.append(f"{weight:.2f} {name}")
            texts.append(" + ".join(parts) if parts else self.domain_names[int(row.argmax())])
        return texts

__all__ = ["SourceStyleBank"]
