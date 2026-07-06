import torch
import torch.nn.functional as F


def domain_uncertainty_loss(style_logits: torch.Tensor):
    log_probs = F.log_softmax(style_logits, dim=-1)
    probs = log_probs.exp()
    return (probs * log_probs).sum(dim=-1).mean()


def style_similarity_logits(
    features: torch.Tensor,
    style_features: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    features = F.normalize(features, dim=-1)
    style_features = F.normalize(style_features, dim=-1)
    return (features @ style_features.t()) * float(scale)


__all__ = ["domain_uncertainty_loss", "style_similarity_logits"]
