import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def _as_bool(value: str | bool | int | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_int(value: str | int | None, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _as_float(value: str | float | None, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _as_csv_list(value: str | None, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if not value:
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _as_int_tuple3(value: str | tuple[int, int, int] | None, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if value is None:
        return default
    if isinstance(value, tuple):
        parts = value
    else:
        parts = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(parts) != 3:
        raise ValueError(f"Expected three comma-separated integers, got {value!r}")
    return tuple(int(item) for item in parts)


def _default_csv() -> str:
    root = Path(__file__).resolve().parents[1]
    candidate = root / "splits_csv" / "patches_with_prompt.csv"
    return str(candidate)


@dataclass
class Config:
    seed: int = field(default_factory=lambda: _as_int(os.getenv("GENERALIZATION_SEED"), 1337))
    batch_size: int = field(default_factory=lambda: _as_int(os.getenv("GENERALIZATION_BATCH_SIZE"), 128))
    num_workers: int = field(default_factory=lambda: _as_int(os.getenv("GENERALIZATION_NUM_WORKERS"), 0))
    weight_decay: float = field(default_factory=lambda: _as_float(os.getenv("GENERALIZATION_WEIGHT_DECAY"), 0.0))

    train_path: str = field(default_factory=lambda: os.getenv("GENERALIZATION_TRAIN_CSV", _default_csv()))
    val_path: str = field(default_factory=lambda: os.getenv("GENERALIZATION_VAL_CSV", _default_csv()))
    test_ood_path: str = field(default_factory=lambda: os.getenv("GENERALIZATION_TEST_CSV", _default_csv()))
    patches_folder: str = field(
        default_factory=lambda: os.getenv("GENERALIZATION_PATCH_ROOT", "data/patches")
    )
    output_root: str = field(
        default_factory=lambda: os.getenv("GENERALIZATION_OUTPUT_ROOT", "/tmp/generalization_runs")
    )

    input_dims: tuple[int, int, int] = field(
        default_factory=lambda: _as_int_tuple3(os.getenv("GENERALIZATION_INPUT_DIMS"), (32, 32, 32))
    )
    intensity_normalization: str = field(
        default_factory=lambda: os.getenv("GENERALIZATION_INTENSITY_NORMALIZATION", "rescale")
    )
    intensity_percentile_low: float = field(
        default_factory=lambda: _as_float(os.getenv("GENERALIZATION_INTENSITY_PERCENTILE_LOW"), 0.5)
    )
    intensity_percentile_high: float = field(
        default_factory=lambda: _as_float(os.getenv("GENERALIZATION_INTENSITY_PERCENTILE_HIGH"), 99.5)
    )
    use_image_augmentation: bool = field(
        default_factory=lambda: _as_bool(os.getenv("GENERALIZATION_USE_IMAGE_AUGMENTATION"), True)
    )
    augmentation_affine_p: float = 1.0
    augmentation_affine_scales: tuple[float, float] = (0.9, 1.1)
    augmentation_affine_degrees: float = 10.0
    augmentation_affine_translation: float = 5.0
    augmentation_pad_value: float = -1000.0
    augmentation_flip_p: float = 0.5
    augmentation_blur_p: float = 1.0
    augmentation_blur_std: tuple[float, float] = (0.5, 1.5)
    augmentation_noise_p: float = 1.0
    augmentation_noise_std: tuple[float, float] = (10.0, 50.0)
    augmentation_elastic_p: float = 1.0
    augmentation_elastic_sigma: float = 20.0
    augmentation_elastic_points: int = 3
    augmentation_intensity_scale: tuple[float, float] = (0.9, 1.1)
    input_channels: int = 1
    num_classes: int = 1
    model_to_be_used: str = field(default_factory=lambda: os.getenv("GENERALIZATION_MODEL", "resnet50"))
    domain_column: str = field(default_factory=lambda: os.getenv("GENERALIZATION_DOMAIN_COLUMN", "manufacturer"))
    label_column: str = field(default_factory=lambda: os.getenv("GENERALIZATION_LABEL_COLUMN", "label"))
    patch_column: str = field(default_factory=lambda: os.getenv("GENERALIZATION_PATCH_COLUMN", "patch_name"))
    domain_aware_batching: bool = field(
        default_factory=lambda: _as_bool(os.getenv("GENERALIZATION_DOMAIN_AWARE_BATCHING"), True)
    )

    criterion_alpha: float = field(default_factory=lambda: _as_float(os.getenv("GENERALIZATION_FOCAL_ALPHA"), 0.8))
    criterion_gamma: float = field(default_factory=lambda: _as_float(os.getenv("GENERALIZATION_FOCAL_GAMMA"), 2.0))

    use_mixstyle: bool = field(default_factory=lambda: _as_bool(os.getenv("GENERALIZATION_USE_MIXSTYLE"), True))
    mixstyle_p: float = 0.2
    mixstyle_alpha: float = 0.5
    mixstyle_mix: str = "random"

    medstyle_latent_dim: int = field(default_factory=lambda: _as_int(os.getenv("GENERALIZATION_MEDSTYLE_LATENT_DIM"), 512))
    medstyle_text_model_name: str = field(
        default_factory=lambda: os.getenv("GENERALIZATION_TEXT_MODEL_NAME", "microsoft/BiomedVLP-CXR-BERT-specialized")
    )
    medstyle_text_model_path: str | None = field(default_factory=lambda: os.getenv("GENERALIZATION_TEXT_MODEL_PATH"))
    medstyle_phase1_checkpoint: str | None = field(default_factory=lambda: os.getenv("GENERALIZATION_PHASE1_CHECKPOINT"))
    medstyle_trust_remote_code: bool = field(
        default_factory=lambda: _as_bool(os.getenv("GENERALIZATION_TRUST_REMOTE_CODE"), False)
    )
    medstyle_freeze_encoders: bool = True
    medstyle_temperature_init: float = field(default_factory=lambda: _as_float(os.getenv("GENERALIZATION_TEMPERATURE_INIT"), 0.07))
    medstyle_temperature_min: float = field(default_factory=lambda: _as_float(os.getenv("GENERALIZATION_TEMPERATURE_MIN"), 0.01))
    medstyle_alignment_epochs: int = field(default_factory=lambda: _as_int(os.getenv("GENERALIZATION_ALIGNMENT_EPOCHS"), 100))
    medstyle_alignment_batch_size: int = field(
        default_factory=lambda: _as_int(os.getenv("GENERALIZATION_ALIGNMENT_BATCH_SIZE"), 128)
    )
    medstyle_alignment_lr: float = field(default_factory=lambda: _as_float(os.getenv("GENERALIZATION_ALIGNMENT_LR"), 1e-4))
    medstyle_alignment_weight_decay: float = field(
        default_factory=lambda: _as_float(os.getenv("GENERALIZATION_ALIGNMENT_WEIGHT_DECAY"), 0.01)
    )
    medstyle_alignment_warmup_epochs: int = 5
    prompt_mode: str = field(default_factory=lambda: os.getenv("GENERALIZATION_PROMPT_MODE", "mixed"))
    prompt_augmentation: bool = field(default_factory=lambda: _as_bool(os.getenv("GENERALIZATION_PROMPT_AUGMENTATION"), True))
    texture_column: str = field(default_factory=lambda: os.getenv("GENERALIZATION_TEXTURE_COLUMN", "texture"))
    margin_column: str = field(default_factory=lambda: os.getenv("GENERALIZATION_MARGIN_COLUMN", "margin"))
    calcification_column: str = field(default_factory=lambda: os.getenv("GENERALIZATION_CALCIFICATION_COLUMN", "calcification"))

    phase1_epochs: int = 300
    phase1_optimizer: str = "sgd"
    phase1_lr: float = 1e-3
    phase1_scheduler_step_size: int = 50
    phase1_scheduler_gamma: float = 0.1
    phase1_batch_size: int = 128
    swad_article_converge_window: int = 5
    swad_article_tolerance_window: int = 3
    swad_article_tolerance_ratio: float = 0.05

    style_k: int = field(default_factory=lambda: _as_int(os.getenv("GENERALIZATION_STYLE_K"), 20))
    style_beta: float = field(default_factory=lambda: _as_float(os.getenv("GENERALIZATION_STYLE_BETA"), 0.1))
    style_se_reduction: int = 16
    style_arcface_scale: float = 5.0
    style_arcface_margin: float = 0.5
    style_epochs: int = field(default_factory=lambda: _as_int(os.getenv("GENERALIZATION_STYLE_EPOCHS"), 100))
    style_batch_size: int = field(default_factory=lambda: _as_int(os.getenv("GENERALIZATION_STYLE_BATCH_SIZE"), 128))
    phase3_soup_size: int = field(default_factory=lambda: _as_int(os.getenv("GENERALIZATION_PHASE3_SOUP_SIZE"), 5))
    style_lr: float = field(default_factory=lambda: _as_float(os.getenv("GENERALIZATION_STYLE_LR"), 0.008))
    style_momentum: float = 0.9
    style_uncertainty_weight: float = field(default_factory=lambda: _as_float(os.getenv("GENERALIZATION_STYLE_UNCERTAINTY_WEIGHT"), 1.0))
    style_uncertainty_logit_scale: float = field(
        default_factory=lambda: _as_float(os.getenv("GENERALIZATION_STYLE_UNCERTAINTY_LOGIT_SCALE"), 100.0)
    )
    source_domain_names: tuple[str, ...] = field(
        default_factory=lambda: _as_csv_list(os.getenv("GENERALIZATION_SOURCE_DOMAINS"))
    )

    def __post_init__(self) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        method = self.enabled_method_name()
        self.writer_name = f"{stamp}_{method}_{self.model_to_be_used}_seed={self.seed}"
        self.tensorboard_dir = str(Path(self.output_root) / "tensorboard")
        self.saving_directory = str(Path(self.output_root) / "models")
        self.writer_full_path = str(Path(self.tensorboard_dir) / self.writer_name)
        self.save_output_directory = str(Path(self.saving_directory) / self.writer_name)
        self.save_for_models = str(Path(self.save_output_directory) / "models")
        self.configs_folder = self.save_output_directory
        self.logging_data = str(Path(self.save_output_directory) / "training.log")
        self.path_to_predictions_for_metrics = str(Path(self.save_output_directory) / "validation_results.csv")

    def enabled_method_name(self) -> str:
        return "article"

    def initialize_directories(self) -> None:
        for path in (self.tensorboard_dir, self.saving_directory, self.save_output_directory, self.save_for_models):
            Path(path).mkdir(parents=True, exist_ok=True)

    def save_extended_config(self, model=None, optimizer=None, scheduler=None, initial: int = 1) -> None:
        self.initialize_directories()
        name = "config_initial.txt" if initial else "config_final.txt"
        path = Path(self.configs_folder) / name
        with path.open("w") as handle:
            for key, value in sorted(self.__dict__.items()):
                handle.write(f"{key}: {value}\n")
            if model is not None:
                handle.write(f"\nmodel: {model}\n")
            if optimizer is not None:
                handle.write(f"\noptimizer: {optimizer}\n")
            if scheduler is not None:
                handle.write(f"\nscheduler: {scheduler}\n")

    @classmethod
    def from_env(cls, **overrides):
        config = cls()
        for key, value in overrides.items():
            setattr(config, key, value)
        config.__post_init__()
        return config

    @staticmethod
    def load_config(filepath: str):
        config = Config()
        with open(filepath) as handle:
            for line in handle:
                if ": " not in line:
                    continue
                key, value = line.rstrip("\n").split(": ", 1)
                setattr(config, key, value)
        config.__post_init__()
        return config
