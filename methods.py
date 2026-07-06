from __future__ import annotations

from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.optim.swa_utils import update_bn


class MixStyle3D(nn.Module):
    def __init__(self, p: float = 0.2, alpha: float = 0.5, eps: float = 1e-6, mix: str = "random"):
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.mix = mix
        self._activated = True

    def set_activation_status(self, status: bool = True) -> None:
        self._activated = status

    def update_mix_method(self, mix: str = "random") -> None:
        self.mix = mix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or not self.training or not self._activated:
            return x
        if torch.rand((), device=x.device).item() > self.p or x.shape[0] < 2:
            return x

        dims = (2, 3, 4)
        mu = x.mean(dim=dims, keepdim=True).detach()
        sigma = (x.var(dim=dims, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
        x_norm = (x - mu) / sigma

        lam = self.beta.sample((x.shape[0], 1, 1, 1, 1)).to(x.device)
        if self.mix == "crossdomain":
            perm = torch.arange(x.shape[0] - 1, -1, -1, device=x.device)
        else:
            perm = torch.randperm(x.shape[0], device=x.device)

        mixed_mu = lam * mu + (1.0 - lam) * mu[perm]
        mixed_sigma = lam * sigma + (1.0 - lam) * sigma[perm]
        return x_norm * mixed_sigma + mixed_mu


def set_mixstyle_active(model: nn.Module, active: bool) -> None:
    for module in model.modules():
        if isinstance(module, MixStyle3D):
            module.set_activation_status(active)


def recompute_batchnorm(loader, model: nn.Module, device=None) -> None:
    was_training = model.training
    mixstyle_states = [
        (module, module._activated)
        for module in model.modules()
        if isinstance(module, MixStyle3D)
    ]
    try:
        set_mixstyle_active(model, False)
        update_bn(loader, model, device=device)
    finally:
        for module, state in mixstyle_states:
            module.set_activation_status(state)
        model.train(was_training)


class AveragedModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.module = deepcopy(model)
        self.n_averaged = 0
        self.start_step: int | None = None
        self.end_step: int | None = None
        self.end_loss: float | None = None

    @torch.no_grad()
    def update_parameters(self, model: nn.Module, *, start_step: int | None = None, end_step: int | None = None) -> None:
        self.n_averaged += 1
        for avg_param, param in zip(self.module.parameters(), model.parameters()):
            avg_param.data.add_((param.data - avg_param.data) / float(self.n_averaged))
        for avg_buffer, buffer in zip(self.module.buffers(), model.buffers()):
            avg_buffer.copy_(buffer)
        if start_step is not None and self.start_step is None:
            self.start_step = int(start_step)
        if end_step is not None:
            self.end_step = int(end_step)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class DenseSWAD:
    def __init__(
        self,
        model: nn.Module,
        converge_window: int = 3,
        tolerance_window: int = 6,
        tolerance_ratio: float = 0.05,
    ):
        self.averaged_model = AveragedModel(model)
        self.converge_window = max(1, int(converge_window))
        self.tolerance_window = max(1, int(tolerance_window))
        self.tolerance_ratio = float(tolerance_ratio)
        self.converge_queue = deque(maxlen=self.converge_window)
        self.smooth_queue = deque(maxlen=self.tolerance_window)
        self.current_segment = AveragedModel(model)
        self.threshold: float | None = None
        self.converge_step: int | None = None
        self.start_step: int | None = None
        self.end_step: int | None = None
        self.active = False
        self.stopped = False
        self.dead_valley = False

    @property
    def n_averaged(self) -> int:
        return int(self.averaged_model.n_averaged)

    @property
    def state(self) -> str:
        if self.stopped:
            return "stopped"
        if self.active:
            return "averaging"
        return "waiting"

    def _new_segment(self, model: nn.Module, step: int | None) -> None:
        self.current_segment = AveragedModel(model)
        self.current_segment.start_step = None if step is None else int(step)

    def _copy_segment(self, segment: AveragedModel) -> None:
        self.averaged_model = AveragedModel(segment.module)
        self.averaged_model.n_averaged = 1
        self.averaged_model.start_step = segment.start_step
        self.averaged_model.end_step = segment.end_step
        self.averaged_model.end_loss = segment.end_loss

    def _append_segment(self, segment: AveragedModel) -> None:
        self.averaged_model.update_parameters(
            segment.module,
            start_step=segment.start_step,
            end_step=segment.end_step,
        )
        self.averaged_model.end_loss = segment.end_loss
        self.start_step = self.averaged_model.start_step
        self.end_step = self.averaged_model.end_step

    def _smooth_min(self) -> float:
        losses = [segment.end_loss for segment in self.smooth_queue if segment.end_loss is not None]
        return float(min(losses)) if losses else float("inf")

    def update_validation(self, validation_loss: float, step: int | None = None) -> bool:
        if self.current_segment.n_averaged == 0:
            return self.active
        loss = float(validation_loss)
        segment = deepcopy(self.current_segment)
        segment.end_loss = loss
        segment.end_step = None if step is None else int(step)
        self.converge_queue.append(segment)
        self.smooth_queue.append(segment)

        if not self.active and not self.stopped:
            if len(self.converge_queue) >= self.converge_window:
                losses = [item.end_loss for item in self.converge_queue]
                min_index = int(np.argmin(losses))
                if min_index == 0:
                    self.active = True
                    self.converge_step = self.converge_queue[0].end_step
                    self.threshold = float(np.mean(losses) * (1.0 + self.tolerance_ratio))
                    self._copy_segment(self.converge_queue[0])
                    if self.tolerance_window < self.converge_window:
                        for item in list(self.converge_queue)[1 : 1 + self.converge_window - self.tolerance_window]:
                            self._append_segment(item)
                    elif self.tolerance_window > self.converge_window:
                        converge_index = self.tolerance_window - self.converge_window
                        queue = list(self.smooth_queue)[: converge_index + 1]
                        start_index = 0
                        for index in reversed(range(len(queue))):
                            item = queue[index]
                            if item.end_loss is not None and item.end_loss > self.threshold:
                                start_index = index + 1
                                break
                        for item in queue[start_index + 1 :]:
                            self._append_segment(item)

        elif self.active and not self.stopped:
            if self.threshold is not None and self._smooth_min() > self.threshold:
                self.active = False
                self.stopped = True
                self.dead_valley = True
                self.end_step = self.averaged_model.end_step
            else:
                item = self.smooth_queue[0]
                if item.end_step is not None and self.converge_step is not None and item.end_step >= self.converge_step:
                    self._append_segment(item)

        self._new_segment(segment.module, step)

        return self.active

    @torch.no_grad()
    def on_train_batch(self, model: nn.Module, step: int | None = None) -> bool:
        if self.current_segment.start_step is None and step is not None:
            self.current_segment.start_step = int(step)
        self.current_segment.update_parameters(model, end_step=step)
        return True

    def finalize(self) -> None:
        if not self.active or self.threshold is None:
            return
        if self.smooth_queue:
            self.smooth_queue.popleft()
        while self.smooth_queue:
            if self._smooth_min() > self.threshold:
                break
            segment = self.smooth_queue.popleft()
            if segment.end_step is None or self.converge_step is None or segment.end_step < self.converge_step:
                continue
            self._append_segment(segment)

    def summary(self) -> dict:
        return {
            "state": self.state,
            "active": self.active,
            "stopped": self.stopped,
            "dead_valley": self.dead_valley,
            "n_averaged": self.n_averaged,
            "threshold": self.threshold,
            "converge_step": self.converge_step,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "converge_window": self.converge_window,
            "tolerance_window": self.tolerance_window,
            "tolerance_ratio": self.tolerance_ratio,
        }
