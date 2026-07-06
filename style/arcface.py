import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceClassifier(nn.Module):
    def __init__(self, embedding_dim: int = 512, num_classes: int = 2, scale: float = 5.0, margin: float = 0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, embedding_dim) * 0.02)
        self.scale = scale
        self.margin = margin

    def forward(self, features: torch.Tensor, labels: torch.Tensor | None = None):
        features = F.normalize(features, dim=-1)
        weights = F.normalize(self.weight, dim=-1)
        cosine = features @ weights.t()
        if labels is None:
            return cosine * self.scale
        labels = labels.long()
        theta = torch.acos(cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
        target = torch.cos(theta + self.margin)
        one_hot = F.one_hot(labels, num_classes=cosine.shape[1]).float()
        logits = cosine * (1.0 - one_hot) + target * one_hot
        return logits * self.scale


def arcface_loss(classifier: ArcFaceClassifier, features: torch.Tensor, labels: torch.Tensor):
    return F.cross_entropy(classifier(features, labels), labels.long())


__all__ = ["ArcFaceClassifier", "arcface_loss"]
