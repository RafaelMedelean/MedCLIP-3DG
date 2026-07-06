from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    import pandas as pd
except Exception:
    pd = None

from .config import PatchPreprocessConfig, parse_tuple


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if pd is not None and pd.isna(value):
        return True
    return isinstance(value, str) and not value.strip()


def _row_value(row, column: str, default: Any = None) -> Any:
    if not column:
        return default
    value = row.get(column, default)
    return default if _missing(value) else value


def _safe_name(value: Any, fallback: str) -> str:
    text = str(value).strip() if not _missing(value) else fallback
    out = []
    for char in text:
        out.append(char if char.isalnum() or char in {"-", "_", "."} else "_")
    return "".join(out).strip("._") or fallback


def _relative_patch_name(row, index: int, cfg: PatchPreprocessConfig) -> str:
    existing = _row_value(row, cfg.patch_column)
    if existing is not None:
        path = Path(str(existing))
        if not path.is_absolute() and ".." not in path.parts:
            return str(path)
        return path.name
    ct_id = _safe_name(_row_value(row, cfg.ct_id_column), f"row_{index:06d}")
    return f"{ct_id}_{index:06d}.npy"


def _find_volume_path(row, cfg: PatchPreprocessConfig, ct_root: str | Path | None) -> Path:
    image_value = _row_value(row, cfg.image_column)
    if image_value is not None:
        path = Path(str(image_value))
        if not path.is_absolute() and ct_root is not None:
            path = Path(ct_root) / path
        return path

    ct_id = _row_value(row, cfg.ct_id_column)
    if ct_id is None or ct_root is None:
        raise ValueError(
            f"Row has no {cfg.image_column!r} path. Provide that column or set ct_root with {cfg.ct_id_column!r}."
        )

    root = Path(ct_root)
    direct = root / str(ct_id)
    if direct.exists():
        return direct
    for suffix in cfg.image_suffixes:
        candidate = root / f"{ct_id}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve volume for ct_id={ct_id!r} under {root}")


def _to_zyx(array: np.ndarray, array_order: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 4:
        if array.shape[0] == 1:
            array = array[0]
        elif array.shape[-1] == 1:
            array = array[..., 0]
        else:
            raise ValueError(f"Expected one channel volume, got shape {array.shape}")
    if array.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {array.shape}")
    if array_order in {"zyx", "dhw"}:
        return np.asarray(array, dtype=np.float32)
    if array_order == "xyz":
        return np.asarray(np.transpose(array, (2, 1, 0)), dtype=np.float32)
    raise ValueError(f"Unsupported array_order={array_order!r}")


def _spacing_from_row(row, cfg: PatchPreprocessConfig) -> tuple[float, float, float] | None:
    z = _row_value(row, cfg.spacing_z_column)
    y = _row_value(row, cfg.spacing_y_column)
    x = _row_value(row, cfg.spacing_x_column)
    if z is None or y is None or x is None:
        return None
    return (float(z), float(y), float(x))


def _load_simpleitk(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    try:
        import SimpleITK as sitk
    except Exception as exc:
        raise ImportError(f"SimpleITK is required to read {path}") from exc
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    spacing_x, spacing_y, spacing_z = image.GetSpacing()
    return array, (float(spacing_z), float(spacing_y), float(spacing_x))


def _find_dicom_files(path: Path) -> list[Path]:
    files = [item for item in path.rglob("*") if item.is_file()]
    dicom_files = [item for item in files if item.suffix.lower() == ".dcm"]
    return sorted(dicom_files or files)


def _load_dicom_directory_simpleitk(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    try:
        import SimpleITK as sitk
    except Exception as exc:
        raise ImportError(f"SimpleITK is required to read DICOM directory {path}") from exc
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(path))
    if series_ids:
        series_files = max(
            (reader.GetGDCMSeriesFileNames(str(path), series_id) for series_id in series_ids),
            key=len,
        )
    else:
        series_files = [str(item) for item in _find_dicom_files(path)]
    if not series_files:
        raise FileNotFoundError(f"No DICOM files under {path}")
    reader.SetFileNames(series_files)
    image = reader.Execute()
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    spacing_x, spacing_y, spacing_z = image.GetSpacing()
    return array, (float(spacing_z), float(spacing_y), float(spacing_x))


def _load_dicom_directory(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    try:
        import pydicom
    except Exception:
        return _load_dicom_directory_simpleitk(path)

    slices = []
    for index, file_path in enumerate(_find_dicom_files(path)):
        try:
            ds = pydicom.dcmread(str(file_path), force=True)
            pixel_array = ds.pixel_array
        except Exception:
            continue
        ipp = getattr(ds, "ImagePositionPatient", None)
        z_pos = float(ipp[2]) if ipp is not None and len(ipp) >= 3 else None
        instance = int(getattr(ds, "InstanceNumber", index))
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        spacing = getattr(ds, "PixelSpacing", None)
        slice_thickness = getattr(ds, "SpacingBetweenSlices", None) or getattr(ds, "SliceThickness", None)
        array = pixel_array.astype(np.float32) * slope + intercept
        slices.append(
            {
                "z_pos": z_pos,
                "instance": instance,
                "array": array,
                "spacing": spacing,
                "slice_thickness": slice_thickness,
            }
        )
    if not slices:
        return _load_dicom_directory_simpleitk(path)

    shapes: dict[tuple[int, int], int] = {}
    for item in slices:
        shapes[item["array"].shape] = shapes.get(item["array"].shape, 0) + 1
    modal_shape = max(shapes.items(), key=lambda item: item[1])[0]
    slices = [item for item in slices if item["array"].shape == modal_shape]
    slices.sort(
        key=lambda item: (
            float(item["z_pos"]) if item["z_pos"] is not None else float(item["instance"]),
            float(item["instance"]),
        )
    )

    array = np.stack([item["array"] for item in slices], axis=0).astype(np.float32, copy=False)
    first = slices[0]
    spacing_yx = first["spacing"]
    if spacing_yx is not None and len(spacing_yx) >= 2:
        spacing_y, spacing_x = float(spacing_yx[0]), float(spacing_yx[1])
    else:
        spacing_y = spacing_x = 1.0
    z_positions = [item["z_pos"] for item in slices if item["z_pos"] is not None]
    if len(z_positions) >= 2:
        diffs = np.diff(np.asarray(z_positions, dtype=np.float64))
        diffs = np.abs(diffs[np.abs(diffs) > 1e-6])
        spacing_z = float(np.median(diffs)) if len(diffs) else 1.0
    elif first["slice_thickness"] is not None:
        spacing_z = float(first["slice_thickness"])
    else:
        spacing_z = 1.0
    return array, (spacing_z, spacing_y, spacing_x)


def _load_nifti(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    try:
        import nibabel as nib
    except Exception:
        return _load_simpleitk(path)
    image = nib.load(str(path))
    array_xyz = image.get_fdata(dtype=np.float32)
    spacing_x, spacing_y, spacing_z = image.header.get_zooms()[:3]
    return np.transpose(array_xyz, (2, 1, 0)), (float(spacing_z), float(spacing_y), float(spacing_x))


def _load_volume(path: Path, row, cfg: PatchPreprocessConfig) -> tuple[np.ndarray, tuple[float, float, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Volume not found: {path}")

    row_spacing = _spacing_from_row(row, cfg)
    name = path.name.lower()
    suffix = "".join(path.suffixes).lower()

    if path.is_dir():
        array, spacing = _load_dicom_directory(path)
        spacing = row_spacing or spacing
    elif suffix == ".npy":
        array = _to_zyx(np.load(path), cfg.array_order)
        spacing = row_spacing or cfg.default_spacing
    elif suffix == ".npz":
        loaded = np.load(path)
        key = cfg.npz_array_key or ("volume" if "volume" in loaded else loaded.files[0])
        array = _to_zyx(loaded[key], cfg.array_order)
        if row_spacing is not None:
            spacing = row_spacing
        elif cfg.npz_spacing_key in loaded:
            spacing = tuple(float(v) for v in np.asarray(loaded[cfg.npz_spacing_key]).reshape(-1)[:3])
        else:
            spacing = cfg.default_spacing
    elif name.endswith(".nii") or name.endswith(".nii.gz"):
        array, spacing = _load_nifti(path)
        spacing = row_spacing or spacing
    else:
        array, spacing = _load_simpleitk(path)
        spacing = row_spacing or spacing

    if len(spacing) != 3:
        raise ValueError(f"Spacing must have three z/y/x values for {path}, got {spacing!r}")
    return np.asarray(array, dtype=np.float32), tuple(float(v) for v in spacing)


def _sanitize_and_clip(volume: np.ndarray, cfg: PatchPreprocessConfig) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32).copy()
    invalid = ~np.isfinite(volume)
    if cfg.invalid_hu_value is not None:
        invalid |= volume < float(cfg.invalid_hu_value)
    if invalid.any():
        volume[invalid] = float(cfg.invalid_hu_replacement)
    if cfg.clip_min is not None or cfg.clip_max is not None:
        lower = -np.inf if cfg.clip_min is None else float(cfg.clip_min)
        upper = np.inf if cfg.clip_max is None else float(cfg.clip_max)
        volume = np.clip(volume, lower, upper)
    return volume.astype(np.float32, copy=False)


def _target_shape(
    volume: np.ndarray,
    source_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
) -> tuple[int, int, int]:
    source = np.asarray(source_spacing, dtype=np.float64)
    target = np.asarray(target_spacing, dtype=np.float64)
    old_shape = np.asarray(volume.shape, dtype=np.float64)
    return tuple(max(1, int(round(size))) for size in old_shape * source / target)


def _actual_spacing(
    volume: np.ndarray,
    source_spacing: tuple[float, float, float],
    new_shape: tuple[int, int, int],
) -> tuple[float, float, float]:
    source = np.asarray(source_spacing, dtype=np.float64)
    old_shape = np.asarray(volume.shape, dtype=np.float64)
    shape = np.asarray(new_shape, dtype=np.float64)
    return tuple((source * old_shape / shape).astype(np.float64).tolist())


def _resample_torch_trilinear(
    volume: np.ndarray,
    source_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
) -> tuple[np.ndarray, tuple[float, float, float]]:
    new_shape = _target_shape(volume, source_spacing, target_spacing)
    if tuple(volume.shape) == tuple(new_shape):
        return volume.astype(np.float32, copy=False), tuple(float(v) for v in source_spacing)
    tensor = torch.from_numpy(np.ascontiguousarray(volume)).float().view(1, 1, *volume.shape)
    with torch.no_grad():
        out = F.interpolate(tensor, size=new_shape, mode="trilinear", align_corners=False)
    return out[0, 0].cpu().numpy().astype(np.float32, copy=False), _actual_spacing(volume, source_spacing, new_shape)


def _resample_dali(
    volume: np.ndarray,
    source_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
    device: str,
    device_id: int,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    try:
        import nvidia.dali.fn as fn
        import nvidia.dali.types as types
        from nvidia.dali.pipeline import Pipeline
    except Exception as exc:
        raise ImportError(
            "NVIDIA DALI is required for article-aligned LANCZOS3 resampling. "
            "Install nvidia-dali-cuda120 or run with --resample-backend torch_trilinear."
        ) from exc
    new_shape = _target_shape(volume, source_spacing, target_spacing)
    if tuple(volume.shape) == tuple(new_shape):
        return volume.astype(np.float32, copy=False), tuple(float(v) for v in source_spacing)
    sample = np.ascontiguousarray(volume[np.newaxis, ...].astype(np.float32, copy=False))

    def source():
        return [sample]

    pipe = Pipeline(batch_size=1, num_threads=3, device_id=int(device_id), exec_pipelined=True, exec_async=True)
    with pipe:
        data = fn.external_source(source=source, device=device, layout="CDHW")
        resized = fn.resize(
            data,
            resize_z=int(new_shape[0]),
            resize_y=int(new_shape[1]),
            resize_x=int(new_shape[2]),
            interp_type=types.INTERP_LANCZOS3,
        )
        pipe.set_outputs(resized)
    pipe.build()
    out = pipe.run()[0].as_cpu().at(0)[0]
    return np.asarray(out, dtype=np.float32), _actual_spacing(volume, source_spacing, new_shape)


def _resample(
    volume: np.ndarray,
    source_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
    cfg: PatchPreprocessConfig,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    if np.allclose(source_spacing, target_spacing, rtol=1e-3, atol=1e-4):
        return volume.astype(np.float32, copy=False), tuple(float(v) for v in source_spacing)
    if cfg.resample_backend == "dali":
        return _resample_dali(volume, source_spacing, target_spacing, cfg.resample_device, cfg.dali_device_id)
    return _resample_torch_trilinear(volume, source_spacing, target_spacing)


def _center_from_row(
    row,
    cfg: PatchPreprocessConfig,
    source_spacing: tuple[float, float, float],
    output_spacing: tuple[float, float, float],
) -> np.ndarray:
    x = float(_row_value(row, cfg.x_column, 0.0))
    y = float(_row_value(row, cfg.y_column, 0.0))
    z = float(_row_value(row, cfg.z_column, 0.0))
    center_zyx = np.asarray((z, y, x), dtype=np.float64)
    if cfg.coordinate_units == "mm":
        return center_zyx / np.asarray(output_spacing, dtype=np.float64)
    return center_zyx * np.asarray(source_spacing, dtype=np.float64) / np.asarray(output_spacing, dtype=np.float64)


def _crop_center(volume: np.ndarray, center_zyx: np.ndarray, shape: tuple[int, int, int], pad_value: float) -> np.ndarray:
    shape_arr = np.asarray(shape, dtype=np.int64)
    start = np.rint(center_zyx).astype(np.int64) - shape_arr // 2
    end = start + shape_arr
    volume_shape = np.asarray(volume.shape, dtype=np.int64)

    src_start = np.maximum(start, 0)
    src_end = np.minimum(end, volume_shape)
    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)

    patch = np.full(tuple(shape_arr), float(pad_value), dtype=np.float32)
    if np.any(src_end <= src_start):
        return patch

    src_slices = tuple(slice(int(a), int(b)) for a, b in zip(src_start, src_end))
    dst_slices = tuple(slice(int(a), int(b)) for a, b in zip(dst_start, dst_end))
    patch[dst_slices] = volume[src_slices]
    return patch


def _normalize(patch: np.ndarray, cfg: PatchPreprocessConfig) -> np.ndarray:
    mode = cfg.normalization
    if mode in {"none", "raw", "hu"}:
        return patch.astype(np.float32, copy=False)
    if mode in {"zscore", "standardize"}:
        mean = float(np.mean(patch))
        std = float(np.std(patch))
        if std < 1e-6:
            return (patch - mean).astype(np.float32, copy=False)
        return ((patch - mean) / std).astype(np.float32, copy=False)
    if mode in {"minmax", "window"}:
        lower = float(cfg.clip_min) if cfg.clip_min is not None else float(np.min(patch))
        upper = float(cfg.clip_max) if cfg.clip_max is not None else float(np.max(patch))
        scale = max(upper - lower, 1e-6)
        return ((patch - lower) / scale).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported normalization={cfg.normalization!r}")


def preprocess_dataframe(
    frame,
    patch_root: str | Path,
    cfg: PatchPreprocessConfig | None = None,
    ct_root: str | Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    if pd is None:
        raise ImportError("pandas is required for preprocessing CSV files")
    cfg = cfg or PatchPreprocessConfig()
    patch_root = Path(patch_root)
    patch_root.mkdir(parents=True, exist_ok=True)

    frame = frame.copy()
    written = 0
    for index, row in frame.iterrows():
        volume_path = _find_volume_path(row, cfg, ct_root)
        patch_name = _relative_patch_name(row, int(index), cfg)
        output_path = patch_root / patch_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and not cfg.overwrite:
            frame.at[index, cfg.patch_column] = patch_name
            continue

        volume, source_spacing = _load_volume(volume_path, row, cfg)
        volume = _sanitize_and_clip(volume, cfg)
        volume, actual_spacing = _resample(volume, source_spacing, cfg.target_spacing, cfg)
        volume = _sanitize_and_clip(volume, cfg)
        center = _center_from_row(row, cfg, source_spacing, actual_spacing)
        patch = _crop_center(volume, center, cfg.output_patch_shape, cfg.pad_value)
        patch = _normalize(patch, cfg)
        np.save(output_path, patch.astype(np.float32, copy=False))

        frame.at[index, cfg.patch_column] = patch_name
        frame.at[index, "source_image_path"] = str(volume_path)
        frame.at[index, "preprocess_spacing_z"] = actual_spacing[0]
        frame.at[index, "preprocess_spacing_y"] = actual_spacing[1]
        frame.at[index, "preprocess_spacing_x"] = actual_spacing[2]
        frame.at[index, "preprocess_shape_z"] = cfg.output_patch_shape[0]
        frame.at[index, "preprocess_shape_y"] = cfg.output_patch_shape[1]
        frame.at[index, "preprocess_shape_x"] = cfg.output_patch_shape[2]
        written += 1

    summary = {
        "rows": int(len(frame)),
        "written": int(written),
        "patch_root": str(patch_root),
        "config": cfg.to_dict(),
    }
    return frame, summary


def preprocess_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    patch_root: str | Path,
    cfg: PatchPreprocessConfig | None = None,
    ct_root: str | Path | None = None,
) -> dict[str, Any]:
    if pd is None:
        raise ImportError("pandas is required for preprocessing CSV files")
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    frame = pd.read_csv(input_csv, comment="#")
    out_frame, summary = preprocess_dataframe(frame, patch_root=patch_root, cfg=cfg, ct_root=ct_root)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_frame.to_csv(output_csv, index=False)
    manifest_path = output_csv.with_suffix(output_csv.suffix + ".manifest.json")
    with manifest_path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    summary["output_csv"] = str(output_csv)
    summary["manifest"] = str(manifest_path)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize resampled nodule patches from CT volumes.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--patch-root", required=True)
    parser.add_argument("--ct-root")
    parser.add_argument("--patch-shape", default="64,45,45", help="Output patch shape in z,y,x order.")
    parser.add_argument("--target-spacing", default="1,1,1", help="Target spacing in z,y,x order.")
    parser.add_argument("--default-spacing", default="1,1,1", help="Fallback source spacing in z,y,x order.")
    parser.add_argument("--resample-backend", default="dali", choices=["dali", "torch_trilinear"])
    parser.add_argument("--resample-device", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--dali-device-id", type=int, default=0)
    parser.add_argument("--normalization", default="none", choices=["none", "raw", "hu", "zscore", "standardize", "minmax", "window"])
    parser.add_argument("--clip-min", type=float, default=-1350.0)
    parser.add_argument("--clip-max", type=float, default=150.0)
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--pad-value", type=float, default=-1000.0)
    parser.add_argument("--image-column", default="image_path")
    parser.add_argument("--ct-id-column", default="ct_id")
    parser.add_argument("--patch-column", default="patch_name")
    parser.add_argument("--x-column", default="x")
    parser.add_argument("--y-column", default="y")
    parser.add_argument("--z-column", default="z")
    parser.add_argument("--coordinate-units", default="voxel", choices=["voxel", "mm"])
    parser.add_argument("--array-order", default="zyx", choices=["zyx", "dhw", "xyz"])
    parser.add_argument("--no-overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = PatchPreprocessConfig(
        image_column=args.image_column,
        ct_id_column=args.ct_id_column,
        patch_column=args.patch_column,
        x_column=args.x_column,
        y_column=args.y_column,
        z_column=args.z_column,
        coordinate_units=args.coordinate_units,
        output_patch_shape=parse_tuple(args.patch_shape, int, "patch_shape"),
        target_spacing=parse_tuple(args.target_spacing, float, "target_spacing"),
        default_spacing=parse_tuple(args.default_spacing, float, "default_spacing"),
        resample_backend=args.resample_backend,
        resample_device=args.resample_device,
        dali_device_id=args.dali_device_id,
        array_order=args.array_order,
        clip_min=None if args.no_clip else args.clip_min,
        clip_max=None if args.no_clip else args.clip_max,
        normalization=args.normalization,
        pad_value=args.pad_value,
        overwrite=not args.no_overwrite,
    )
    summary = preprocess_csv(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        patch_root=args.patch_root,
        cfg=cfg,
        ct_root=args.ct_root,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
