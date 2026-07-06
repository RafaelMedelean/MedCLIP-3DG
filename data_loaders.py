from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from configs.config import Config
except Exception:
    Config = None


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def custom_collate_fn(batch):
    images, labels, domains, metadata = zip(*batch)
    meta_keys = metadata[0].keys() if metadata else []
    collated = {key: [m[key] for m in metadata] for key in meta_keys}
    return torch.stack(images), torch.tensor(labels).float(), torch.tensor(domains).long(), collated


def _stable_domain_id(value: object, mapping: dict[str, int]) -> int:
    is_missing = pd.isna(value) if pd is not None else value is None
    key = "unknown" if is_missing else str(value)
    if key not in mapping:
        mapping[key] = len(mapping)
    return mapping[key]


def _tuple3(value: Iterable[float | int]) -> tuple:
    values = tuple(value)
    if len(values) != 3:
        raise ValueError(f"Expected three values, got {value!r}")
    return values


def _tuple2(value: Iterable[float | int]) -> tuple:
    values = tuple(value)
    if len(values) != 2:
        raise ValueError(f"Expected two values, got {value!r}")
    return values


class DomainAwareBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset, batch_size: int, seed: int = 2022):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        frame = getattr(dataset, "frame", None)
        config = getattr(dataset, "config", None)
        if frame is None or config is None:
            raise ValueError("DomainAwareBatchSampler requires a dataset with frame and config.")
        column = config.domain_column
        if column not in frame.columns:
            domains = ["unknown"] * len(frame)
        else:
            domains = frame[column].fillna("unknown").astype(str).tolist()
        self.domain_to_indices: dict[str, list[int]] = {}
        for index, domain in enumerate(domains):
            self.domain_to_indices.setdefault(domain, []).append(index)
        self.domains = sorted(self.domain_to_indices)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self.domains:
            raise ValueError("DomainAwareBatchSampler requires at least one sample")

    def __len__(self) -> int:
        return max(1, int(np.ceil(len(self.dataset) / self.batch_size)))

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        buckets = {domain: list(indices) for domain, indices in self.domain_to_indices.items()}
        pointers = {domain: 0 for domain in self.domains}
        for indices in buckets.values():
            rng.shuffle(indices)

        def next_index(domain: str) -> int:
            if pointers[domain] >= len(buckets[domain]):
                rng.shuffle(buckets[domain])
                pointers[domain] = 0
            index = buckets[domain][pointers[domain]]
            pointers[domain] += 1
            return index

        for _ in range(len(self)):
            order = list(self.domains)
            rng.shuffle(order)
            if self.batch_size >= len(order):
                selected = order[:]
            else:
                selected = order[: self.batch_size]
            batch = [next_index(domain) for domain in selected]
            fill_order = order[:]
            while len(batch) < self.batch_size:
                if not fill_order:
                    fill_order = list(self.domains)
                    rng.shuffle(fill_order)
                batch.append(next_index(fill_order.pop(0)))
            rng.shuffle(batch)
            yield batch


def _elastic_deformation_transform(config, tio):
    try:
        import elasticdeform
    except Exception as exc:
        raise ImportError(
            "elasticdeform is required for the elastic-deformation augmentation. Install it with `pip install elasticdeform`."
        ) from exc

    sigma = float(config.augmentation_elastic_sigma)
    points = int(config.augmentation_elastic_points)
    cval = float(config.augmentation_pad_value)
    probability = float(config.augmentation_elastic_p)

    def deform(tensor: torch.Tensor) -> torch.Tensor:
        if probability < 1.0 and random.random() >= probability:
            return tensor
        array = tensor.detach().cpu().numpy()
        channels = [array[channel] for channel in range(array.shape[0])]
        deformed = elasticdeform.deform_random_grid(
            channels, sigma=sigma, points=points, order=3, mode="constant", cval=cval
        )
        stacked = np.stack(deformed, axis=0).astype(np.float32, copy=False)
        return torch.from_numpy(np.ascontiguousarray(stacked)).to(tensor.dtype)

    return tio.Lambda(deform)


def _build_torchio_transform(config, is_train: bool):
    try:
        import torchio as tio
    except Exception as exc:
        raise ImportError("TorchIO is required for patch preprocessing/augmentation. Install it with `pip install torchio`.") from exc

    dims = _tuple3(config.input_dims)
    augment = is_train and getattr(config, "use_image_augmentation", True)
    transforms = []
    if augment:
        transforms.append(
            tio.RandomAffine(
                scales=_tuple2(config.augmentation_affine_scales),
                degrees=float(config.augmentation_affine_degrees),
                translation=float(config.augmentation_affine_translation),
                default_pad_value=float(config.augmentation_pad_value),
                p=float(config.augmentation_affine_p),
            )
        )
        transforms.append(_elastic_deformation_transform(config, tio))
    transforms.append(tio.CropOrPad(target_shape=dims, padding_mode=float(config.augmentation_pad_value)))
    if augment:
        transforms.append(
            tio.RandomNoise(
                mean=0.0,
                std=_tuple2(config.augmentation_noise_std),
                p=float(config.augmentation_noise_p),
            )
        )
    normalization = str(getattr(config, "intensity_normalization", "rescale")).lower()
    if normalization in {"rescale", "rescale_intensity", "percentile"}:
        transforms.append(
            tio.RescaleIntensity(
                out_min_max=(0, 1),
                percentiles=(float(config.intensity_percentile_low), float(config.intensity_percentile_high)),
            )
        )
    elif normalization not in {"none", "raw", "zscore", "standardize"}:
        raise ValueError(f"Unsupported intensity_normalization={normalization!r}")
    if augment:
        transforms.append(tio.RandomFlip(axes=(2,), p=float(config.augmentation_flip_p)))
        transforms.append(tio.RandomBlur(std=_tuple2(config.augmentation_blur_std), p=float(config.augmentation_blur_p)))
    return tio.Compose(transforms)


def _apply_torchio_transform(tensor: torch.Tensor, transform) -> torch.Tensor:
    import torchio as tio

    subject = tio.Subject(image=tio.ScalarImage(tensor=tensor))
    transformed = transform(subject)
    return transformed.image.data.float()


class NodulePatchDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        patch_root: str,
        config,
        domain_mapping: dict[str, int] | None = None,
        is_train: bool = False,
    ):
        self.csv_path = Path(csv_path)
        self.patch_root = Path(patch_root)
        self.config = config
        self.domain_mapping = domain_mapping if domain_mapping is not None else {}
        self.is_train = bool(is_train)
        self.transform = _build_torchio_transform(config, self.is_train)
        if pd is None:
            raise ImportError("pandas is required for CSV loading.")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        self.frame = pd.read_csv(self.csv_path, comment="#")
        if self.config.label_column not in self.frame.columns:
            raise ValueError(f"Missing label column {self.config.label_column!r} in {self.csv_path}")
        if self.config.patch_column not in self.frame.columns:
            raise ValueError(f"Missing patch column {self.config.patch_column!r} in {self.csv_path}")
        if self.config.domain_column not in self.frame.columns:
            self.frame[self.config.domain_column] = "unknown"

    def __len__(self) -> int:
        return len(self.frame)

    def _load_patch(self, patch_name: str) -> torch.Tensor:
        path = self.patch_root / str(patch_name)
        if not path.exists():
            raise FileNotFoundError(
                f"Patch file not found: {path}. Set GENERALIZATION_PATCH_ROOT to the folder with .npy patches "
                "that matches the configured CSV files."
            )
        array = np.load(path)
        tensor = torch.as_tensor(array, dtype=torch.float32)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 4 and tensor.shape[0] != 1:
            tensor = tensor[:1]
        if tensor.ndim != 4:
            raise ValueError(f"Expected 3D patch or [C,D,H,W] patch in {path}, got shape {tuple(tensor.shape)}")
        tensor = _apply_torchio_transform(tensor, self.transform)
        if self.is_train and getattr(self.config, "use_image_augmentation", True):
            low, high = _tuple2(self.config.augmentation_intensity_scale)
            scale = torch.empty((), dtype=tensor.dtype).uniform_(float(low), float(high))
            tensor = tensor * scale
        normalization = str(getattr(self.config, "intensity_normalization", "rescale")).lower()
        if normalization in {"zscore", "standardize"}:
            mean = tensor.mean()
            std = tensor.std().clamp_min(1e-6)
            tensor = (tensor - mean) / std
        return tensor.float()

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        label = float(row[self.config.label_column])
        patch_name = str(row[self.config.patch_column])
        domain = _stable_domain_id(row[self.config.domain_column], self.domain_mapping)
        image = self._load_patch(patch_name)
        metadata = {
            "ct_id": row.get("ct_id", ""),
            "z": float(row.get("z", 0.0) or 0.0),
            "y": float(row.get("y", 0.0) or 0.0),
            "x": float(row.get("x", 0.0) or 0.0),
            "length": float(row.get("length", 0.0) or 0.0),
            "patch_name": patch_name,
        }
        return image, label, domain, metadata


def _loader(dataset, config, shuffle: bool):
    generator = torch.Generator().manual_seed(config.seed)
    if shuffle and getattr(config, "domain_aware_batching", True):
        return DataLoader(
            dataset,
            batch_sampler=DomainAwareBatchSampler(dataset, config.batch_size, seed=config.seed),
            num_workers=config.num_workers,
            worker_init_fn=seed_worker,
            collate_fn=custom_collate_fn,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        worker_init_fn=seed_worker,
        collate_fn=custom_collate_fn,
        generator=generator,
        drop_last=False,
    )


def make_data_loaders(config=None):
    config = config or Config()
    domain_mapping: dict[str, int] = {}
    train = NodulePatchDataset(config.train_path, config.patches_folder, config, domain_mapping, is_train=True)
    val = NodulePatchDataset(config.val_path, config.patches_folder, config, domain_mapping, is_train=False)
    test = NodulePatchDataset(config.test_ood_path, config.patches_folder, config, domain_mapping, is_train=False)
    train_loader = _loader(train, config, shuffle=True)
    val_loader = _loader(val, config, shuffle=False)
    test_loader = _loader(test, config, shuffle=False)
    return train_loader, val_loader, test_loader


__all__ = [
    "DomainAwareBatchSampler",
    "NodulePatchDataset",
    "custom_collate_fn",
    "make_data_loaders",
    "seed_worker",
]
