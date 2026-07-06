from __future__ import annotations

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from configs.config import Config
from data_loaders import make_data_loaders
from losses import SigmoidFocalLoss_gamma_alpha
from methods import DenseSWAD, MixStyle3D
from models import ModelHandler


class ArticleSWAD(DenseSWAD):
    def __init__(
        self,
        model,
        converge_window: int = 5,
        tolerance_window: int = 3,
        tolerance_ratio: float = 0.05,
    ):
        super().__init__(
            model,
            converge_window=converge_window,
            tolerance_window=tolerance_window,
            tolerance_ratio=tolerance_ratio,
        )


class Phase1VisualTrainer:
    def __init__(self, config: Config | None = None, device_id: int = 0):
        self.config = config or Config()
        self.config.batch_size = self.config.phase1_batch_size
        self.config.use_mixstyle = True
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.train_loader, self.val_loader, self.test_loader = make_data_loaders(self.config)
        self.model = ModelHandler(self.config).initialize_model(self.config.model_to_be_used).to(self.device)
        self.criterion = SigmoidFocalLoss_gamma_alpha(
            alpha=self.config.criterion_alpha,
            gamma=self.config.criterion_gamma,
        )
        self.optimizer = SGD(
            self.model.parameters(),
            lr=self.config.phase1_lr,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = StepLR(
            self.optimizer,
            step_size=self.config.phase1_scheduler_step_size,
            gamma=self.config.phase1_scheduler_gamma,
        )
        self.swad = ArticleSWAD(
            self.model,
            converge_window=self.config.swad_article_converge_window,
            tolerance_window=self.config.swad_article_tolerance_window,
            tolerance_ratio=self.config.swad_article_tolerance_ratio,
        )
        self.global_step = 0

    def set_mixstyle_active(self, active: bool) -> None:
        if hasattr(self.model, "set_mixstyle_active"):
            self.model.set_mixstyle_active(active)
        for module in self.model.modules():
            if isinstance(module, MixStyle3D):
                module.set_activation_status(active)

    def train_epoch(self):
        self.model.train()
        self.set_mixstyle_active(True)
        total = 0.0
        n = 0
        for images, labels, _domains, _metadata in self.train_loader:
            images = images.float().to(self.device)
            labels = labels.float().to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            self.global_step += 1
            self.swad.on_train_batch(self.model, step=self.global_step)
            total += float(loss.detach().cpu())
            n += 1
        return total / max(1, n)


__all__ = ["ArticleSWAD", "Phase1VisualTrainer"]
