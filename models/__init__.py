from .resnet50_3d import Bottleneck3D, ResNet3D, resnet50, resnet50_3d


class ModelHandler:
    def __init__(self, config, device_id: int = 0):
        self.config = config
        self.device_id = device_id

    def initialize_model(self, model_name: str | None = None):
        name = (model_name or self.config.model_to_be_used).lower()
        common = {
            "num_classes": self.config.num_classes,
            "in_channels": self.config.input_channels,
            "use_mixstyle": self.config.use_mixstyle,
            "mixstyle_p": self.config.mixstyle_p,
            "mixstyle_alpha": self.config.mixstyle_alpha,
            "mixstyle_mix": self.config.mixstyle_mix,
        }
        if name in {"resnet50", "resnet50_3d", "resnet3d"}:
            return resnet50_3d(**common)
        raise ValueError(f"Unknown model name: {model_name}")


__all__ = ["Bottleneck3D", "ModelHandler", "ResNet3D", "resnet50", "resnet50_3d"]
