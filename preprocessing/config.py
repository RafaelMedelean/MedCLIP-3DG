from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable


def parse_tuple(value: str | tuple | list, cast: Callable[[str], Any], name: str) -> tuple:
    if isinstance(value, str):
        text = value.lower().replace("x", ",").replace(";", ",")
        parts = [part.strip() for part in text.split(",") if part.strip()]
    else:
        parts = list(value)
    if len(parts) != 3:
        raise ValueError(f"{name} must have three values, got {value!r}")
    return tuple(cast(part) for part in parts)


@dataclass
class PatchPreprocessConfig:
    image_column: str = "image_path"
    ct_id_column: str = "ct_id"
    patch_column: str = "patch_name"
    x_column: str = "x"
    y_column: str = "y"
    z_column: str = "z"
    spacing_x_column: str = "spacing_x"
    spacing_y_column: str = "spacing_y"
    spacing_z_column: str = "spacing_z"
    coordinate_units: str = "voxel"

    output_patch_shape: tuple[int, int, int] = (64, 45, 45)
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    default_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    resample_backend: str = "dali"
    resample_device: str = "cpu"
    dali_device_id: int = 0
    array_order: str = "zyx"

    npz_array_key: str | None = None
    npz_spacing_key: str = "spacing"

    invalid_hu_value: float | None = -2000.0
    invalid_hu_replacement: float = 0.0
    clip_min: float | None = -1350.0
    clip_max: float | None = 150.0
    normalization: str = "none"
    pad_value: float = -1000.0

    image_suffixes: tuple[str, ...] = (".npy", ".npz", ".nii.gz", ".nii", ".mha", ".mhd", ".nrrd")
    overwrite: bool = True

    def __post_init__(self) -> None:
        self.output_patch_shape = parse_tuple(self.output_patch_shape, int, "output_patch_shape")
        self.target_spacing = parse_tuple(self.target_spacing, float, "target_spacing")
        self.default_spacing = parse_tuple(self.default_spacing, float, "default_spacing")
        self.coordinate_units = self.coordinate_units.lower()
        self.array_order = self.array_order.lower()
        self.normalization = self.normalization.lower()
        self.resample_backend = self.resample_backend.lower()
        self.resample_device = self.resample_device.lower()

        if self.coordinate_units not in {"voxel", "mm"}:
            raise ValueError("coordinate_units must be 'voxel' or 'mm'")
        if self.array_order not in {"zyx", "dhw", "xyz"}:
            raise ValueError("array_order must be one of: zyx, dhw, xyz")
        if self.resample_backend not in {"dali", "torch_trilinear"}:
            raise ValueError("resample_backend must be one of: dali, torch_trilinear")
        if self.resample_device not in {"cpu", "gpu"}:
            raise ValueError("resample_device must be one of: cpu, gpu")
        if self.normalization not in {"none", "raw", "hu", "zscore", "standardize", "minmax", "window"}:
            raise ValueError("normalization must be one of: none, raw, hu, zscore, standardize, minmax, window")
        if any(value <= 0 for value in self.output_patch_shape):
            raise ValueError("output_patch_shape values must be positive")
        if any(value <= 0 for value in self.target_spacing):
            raise ValueError("target_spacing values must be positive")
        if any(value <= 0 for value in self.default_spacing):
            raise ValueError("default_spacing values must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
