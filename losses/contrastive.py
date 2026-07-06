import math

import torch
import torch.nn.functional as F


def symmetric_infonce_loss(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    log_temperature: torch.Tensor,
    min_temperature: float = 0.01,
):
    image_emb = F.normalize(image_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    logits = image_emb @ text_emb.t() / log_temperature.exp().clamp_min(float(min_temperature))
    labels = torch.arange(image_emb.shape[0], device=image_emb.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return (loss_i + loss_t) / 2.0, logits


def temperature_to_log(value: float) -> float:
    return math.log(float(value))


__all__ = ["symmetric_infonce_loss", "temperature_to_log"]
