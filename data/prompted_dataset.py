from __future__ import annotations

import random

import torch
from torch.utils.data import DataLoader

try:
    import pandas as pd
except Exception:
    pd = None

from data_loaders import DomainAwareBatchSampler, NodulePatchDataset, seed_worker


POSITIVE_SIMPLE = "A lung nodule is present in this CT scan."
NEGATIVE_PROMPT = "A lung nodule is not present in this CT scan."

POSITIVE_STYLE_TEMPLATE = "A lung nodule in a {style} style"
NEGATIVE_STYLE_TEMPLATE = "No lung nodule in a {style} style"


TEXT_AUGMENTATIONS = {
    POSITIVE_SIMPLE: [
        "This CT scan contains a lung nodule.",
        "A pulmonary nodule is visible in this CT scan.",
        "The CT image shows a lung nodule.",
    ],
    NEGATIVE_PROMPT: [
        "No lung nodule is present in this CT scan.",
        "This CT scan does not contain a lung nodule.",
        "The CT image shows no lung nodule.",
    ],
}


def _clean_attribute(value, fallback: str = "unknown") -> str:
    if value is None:
        return fallback
    if pd is not None and pd.isna(value):
        return fallback
    text = str(value).strip()
    return text if text else fallback


def build_article_prompt(label, texture=None, margin=None, calcification=None, mode: str = "complex") -> str:
    label_value = float(label)
    if label_value < 0.5:
        return NEGATIVE_PROMPT
    if mode == "simple":
        return POSITIVE_SIMPLE
    texture = _clean_attribute(texture, "unknown")
    margin = _clean_attribute(margin, "unknown")
    calcification = _clean_attribute(calcification, "unknown")
    return (
        "A lung nodule is present in this CT scan with "
        f"{texture} texture, {margin} margin, and {calcification} calcification."
    )


def augment_prompt(prompt: str, rng: random.Random) -> str:
    variants = TEXT_AUGMENTATIONS.get(prompt)
    if variants and rng.random() < 0.5:
        return rng.choice(variants)
    replacements = [
        ("lung nodule", "pulmonary nodule"),
        ("present in", "visible in"),
        ("CT scan", "chest CT scan"),
    ]
    out = prompt
    for src, dst in replacements:
        if src in out and rng.random() < 0.25:
            out = out.replace(src, dst)
    return out


class PromptedNodulePatchDataset(NodulePatchDataset):
    def __init__(self, *args, prompt_mode: str = "mixed", prompt_augmentation: bool = True, seed: int = 2022, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_mode = prompt_mode
        self.prompt_augmentation = prompt_augmentation
        self.seed = seed

    def _mode_for_index(self, index: int) -> str:
        if self.prompt_mode == "mixed":
            return "simple" if index % 2 == 0 else "complex"
        return self.prompt_mode

    def __getitem__(self, index: int):
        image, label, domain, metadata = super().__getitem__(index)
        row = self.frame.iloc[index]
        prompt = build_article_prompt(
            label,
            row.get(self.config.texture_column),
            row.get(self.config.margin_column),
            row.get(self.config.calcification_column),
            self._mode_for_index(index),
        )
        if self.prompt_augmentation:
            prompt = augment_prompt(prompt, random.Random(self.seed + index))
        metadata = dict(metadata)
        metadata.update(
            {
                "prompt": prompt,
                "texture": _clean_attribute(row.get(self.config.texture_column)),
                "margin": _clean_attribute(row.get(self.config.margin_column)),
                "calcification": _clean_attribute(row.get(self.config.calcification_column)),
            }
        )
        return image, label, domain, prompt, metadata


def prompted_collate_fn(batch):
    images, labels, domains, prompts, metadata = zip(*batch)
    meta_keys = metadata[0].keys() if metadata else []
    collated = {key: [m[key] for m in metadata] for key in meta_keys}
    return torch.stack(images), torch.tensor(labels).float(), torch.tensor(domains).long(), list(prompts), collated


def build_prompted_datasets(config):
    domain_mapping: dict[str, int] = {}
    train = PromptedNodulePatchDataset(
        config.train_path,
        config.patches_folder,
        config,
        domain_mapping,
        prompt_mode=config.prompt_mode,
        prompt_augmentation=config.prompt_augmentation,
        seed=config.seed,
        is_train=True,
    )
    val = PromptedNodulePatchDataset(
        config.val_path,
        config.patches_folder,
        config,
        domain_mapping,
        prompt_mode=config.prompt_mode,
        prompt_augmentation=False,
        seed=config.seed,
        is_train=False,
    )
    test = PromptedNodulePatchDataset(
        config.test_ood_path,
        config.patches_folder,
        config,
        domain_mapping,
        prompt_mode=config.prompt_mode,
        prompt_augmentation=False,
        seed=config.seed,
        is_train=False,
    )
    return train, val, test


def make_prompted_data_loaders(config):
    domain_mapping: dict[str, int] = {}
    train = PromptedNodulePatchDataset(
        config.train_path,
        config.patches_folder,
        config,
        domain_mapping,
        prompt_mode=config.prompt_mode,
        prompt_augmentation=config.prompt_augmentation,
        seed=config.seed,
        is_train=True,
    )
    val = PromptedNodulePatchDataset(
        config.val_path,
        config.patches_folder,
        config,
        domain_mapping,
        prompt_mode=config.prompt_mode,
        prompt_augmentation=False,
        seed=config.seed,
        is_train=False,
    )
    test = PromptedNodulePatchDataset(
        config.test_ood_path,
        config.patches_folder,
        config,
        domain_mapping,
        prompt_mode=config.prompt_mode,
        prompt_augmentation=False,
        seed=config.seed,
        is_train=False,
    )
    generator = torch.Generator().manual_seed(config.seed)

    def loader(dataset, shuffle: bool):
        if shuffle and getattr(config, "domain_aware_batching", True):
            return DataLoader(
                dataset,
                batch_sampler=DomainAwareBatchSampler(dataset, config.batch_size, seed=config.seed),
                num_workers=config.num_workers,
                worker_init_fn=seed_worker,
                collate_fn=prompted_collate_fn,
                generator=generator,
            )
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers,
            worker_init_fn=seed_worker,
            collate_fn=prompted_collate_fn,
            generator=generator,
            drop_last=False,
        )

    return loader(train, True), loader(val, False), loader(test, False)


__all__ = [
    "NEGATIVE_PROMPT",
    "POSITIVE_SIMPLE",
    "PromptedNodulePatchDataset",
    "augment_prompt",
    "build_article_prompt",
    "build_prompted_datasets",
    "make_prompted_data_loaders",
    "prompted_collate_fn",
]
