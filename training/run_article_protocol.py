from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import Config
from data.prompted_dataset import (
    NEGATIVE_PROMPT,
    POSITIVE_SIMPLE,
    make_prompted_data_loaders,
)
from data_loaders import make_data_loaders
from evaluation.grid_search_ensemble import best_f1_threshold, grid_search_weights
from inference.medstyle_3dg_ensemble import MedStyle3DGEnsemble
from methods import recompute_batchnorm
from trainers.phase1_visual import Phase1VisualTrainer
from trainers.phase2_alignment import Phase2AlignmentTrainer
from trainers.phase3_style_removal import Phase3StyleRemovalTrainer


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def fixed_threshold_metrics(labels, probabilities, threshold: float = 0.5) -> dict:
    labels = np.asarray(labels).astype(np.int32)
    probabilities = np.asarray(probabilities).astype(np.float32)
    preds = (probabilities >= threshold).astype(np.int32)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float(0.5 * (recall + specificity)),
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def auroc_score(labels, probabilities) -> float:
    labels = np.asarray(labels).astype(np.int32)
    probabilities = np.asarray(probabilities).astype(np.float64)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(probabilities)
    ranks = np.empty_like(probabilities, dtype=np.float64)
    i = 0
    while i < len(probabilities):
        j = i + 1
        while j < len(probabilities) and probabilities[order[j]] == probabilities[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    pos_rank_sum = float(ranks[labels == 1].sum())
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision_score(labels, probabilities) -> float:
    labels = np.asarray(labels).astype(np.int32)
    probabilities = np.asarray(probabilities).astype(np.float64)
    n_pos = int((labels == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-probabilities)
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    precision = tp / (np.arange(len(sorted_labels)) + 1)
    return float(precision[sorted_labels == 1].sum() / n_pos)


def score_summary(
    labels,
    probabilities,
    *,
    val_threshold: float | None = None,
    include_best_threshold: bool = True,
) -> dict:
    labels = np.asarray(labels).astype(np.int32)
    probabilities = np.asarray(probabilities).astype(np.float32)
    out = {
        "n": int(len(labels)),
        "pos": int((labels == 1).sum()),
        "neg": int((labels == 0).sum()),
        "auroc": auroc_score(labels, probabilities),
        "average_precision": average_precision_score(labels, probabilities),
        "prob_mean": float(np.mean(probabilities)) if len(probabilities) else float("nan"),
        "prob_std": float(np.std(probabilities)) if len(probabilities) else float("nan"),
        "fixed_0_5": fixed_threshold_metrics(labels, probabilities, 0.5),
    }
    if include_best_threshold:
        out["best_threshold_on_this_split"] = best_f1_threshold(labels, probabilities)
    if val_threshold is not None:
        out["at_validation_threshold"] = fixed_threshold_metrics(labels, probabilities, val_threshold)
    for label, name in ((1, "positive"), (0, "negative")):
        subset = probabilities[labels == label]
        out[f"{name}_prob_mean"] = float(np.mean(subset)) if len(subset) else float("nan")
        out[f"{name}_prob_std"] = float(np.std(subset)) if len(subset) else float("nan")
    out["pos_neg_mean_gap"] = out["positive_prob_mean"] - out["negative_prob_mean"]
    return out


def state_to_cpu(module: torch.nn.Module) -> dict:
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def average_state_dicts(state_dicts: list[dict]) -> dict:
    if not state_dicts:
        raise ValueError("state_dicts must not be empty")
    averaged = {}
    for name, tensor in state_dicts[0].items():
        if tensor.is_floating_point() or tensor.is_complex():
            total = tensor.detach().clone()
            for state in state_dicts[1:]:
                total.add_(state[name].to(dtype=total.dtype))
            total.div_(len(state_dicts))
            averaged[name] = total
        else:
            averaged[name] = tensor.detach().clone()
    return averaged


def insert_soup_member(members: list[dict], member: dict, limit: int) -> list[dict]:
    members.append(member)
    members.sort(key=lambda item: (item["score"], item["epoch"]), reverse=True)
    del members[max(1, limit):]
    return members


def soup_member_metadata(members: list[dict]) -> list[dict]:
    return [
        {
            "rank": index + 1,
            "epoch": int(member["epoch"]),
            "score": float(member["score"]),
            "threshold": float(member["threshold"]),
            "val": member["val"],
        }
        for index, member in enumerate(members)
    ]


def read_report(split_dir: Path) -> dict:
    report_path = split_dir / "split_report.json"
    if report_path.exists():
        return json.loads(report_path.read_text())
    return {}


def infer_domain_names(csv_path: Path, domain_column: str) -> tuple[str, ...]:
    frame = pd.read_csv(csv_path, comment="#")
    if domain_column not in frame.columns:
        return ("source",)
    values = [str(v) for v in frame[domain_column].dropna().unique()]
    return tuple(sorted(values)) or ("source",)


def preflight_patch_root(split_dir: Path, patch_root: Path, *, patch_column: str, check_limit: int = 250) -> dict:
    missing = {}
    checked = {}
    for name in ("Train", "Valid", "TestOOD", "TrainPlusTestID"):
        path = split_dir / f"{name}.csv"
        frame = pd.read_csv(path, comment="#")
        patch_names = frame[patch_column].astype(str).tolist()
        if len(patch_names) > check_limit:
            random.Random(2022 + len(name)).shuffle(patch_names)
            patch_names = patch_names[:check_limit]
        absent = [patch for patch in patch_names if not (patch_root / patch).exists()]
        checked[name] = len(patch_names)
        if absent:
            missing[name] = absent[:10]
    return {"patch_root": str(patch_root), "checked": checked, "missing_examples": missing, "ok": not missing}


def make_config(args, split_dir: Path, output_root: Path, train_csv: Path, domain_column: str) -> Config:
    source_domains = infer_domain_names(train_csv, domain_column)
    cfg = Config(
        seed=args.seed,
        train_path=str(train_csv),
        val_path=str(split_dir / "Valid.csv"),
        test_ood_path=str(split_dir / "TestOOD.csv"),
        patches_folder=str(args.patch_root),
        output_root=str(output_root),
        num_workers=args.num_workers,
        model_to_be_used=args.model,
        domain_column=domain_column,
        label_column=args.label_column,
        patch_column=args.patch_column,
        medstyle_text_model_path=str(args.text_model) if args.text_model else None,
        medstyle_trust_remote_code=args.trust_remote_code,
        medstyle_alignment_epochs=args.phase2_epochs,
        medstyle_alignment_batch_size=args.phase2_batch_size,
        medstyle_alignment_lr=args.phase2_lr,
        style_epochs=args.phase3_epochs,
        style_batch_size=args.phase3_batch_size,
        style_k=args.style_k,
        style_lr=args.phase3_lr,
        phase3_soup_size=args.phase3_soup_size,
    )
    cfg.phase1_epochs = args.phase1_epochs
    cfg.phase1_batch_size = args.phase1_batch_size
    cfg.phase1_lr = args.phase1_lr
    cfg.weight_decay = args.weight_decay
    cfg.batch_size = args.batch_size
    cfg.source_domain_names = source_domains
    cfg.__post_init__()
    return cfg


def select_score(metrics: dict, metric: str) -> float:
    if metric in {"f1", "fixed_f1", "f1_at_0_5"}:
        value = metrics["fixed_0_5"]["f1"]
    elif metric in {"best_f1", "threshold_f1"}:
        value = metrics["best_threshold_on_this_split"]["f1"]
    else:
        value = metrics.get(metric)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            value = metrics["best_threshold_on_this_split"]["f1"]
    return float(value)


def print_epoch_progress(regimen: str, phase: str, epoch: int, train_loss: float, val: dict) -> None:
    val_f1 = val["fixed_0_5"]["f1"]
    val_best_f1 = val["best_threshold_on_this_split"]["f1"]
    val_threshold = val["best_threshold_on_this_split"]["threshold"]
    print(
        f"[{regimen}] {phase} epoch={epoch} train_loss={train_loss:.6f} "
        f"val_auc={val['auroc']:.4f} val_f1@0.5={val_f1:.4f} val_best_f1={val_best_f1:.4f} "
        f"val_thr={val_threshold:.4f}",
        flush=True,
    )


@torch.no_grad()
def eval_phase1(
    trainer: Phase1VisualTrainer,
    loader,
    *,
    val_threshold: float | None = None,
    model: torch.nn.Module | None = None,
    include_best_threshold: bool = True,
) -> dict:
    eval_model = model or trainer.model
    eval_model.eval()
    if model is None:
        trainer.set_mixstyle_active(False)
    labels = []
    probs = []
    losses = []
    for images, batch_labels, _domains, _metadata in loader:
        images = images.float().to(trainer.device)
        labels_tensor = batch_labels.float().to(trainer.device)
        logits = eval_model(images)
        losses.append(float(trainer.criterion(logits, labels_tensor).detach().cpu()))
        labels.extend(batch_labels.detach().cpu().long().numpy().tolist())
        probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
    out = score_summary(labels, probs, val_threshold=val_threshold, include_best_threshold=include_best_threshold)
    out["loss"] = float(sum(losses) / max(1, len(losses)))
    return out


def run_phase1(
    cfg: Config,
    output_dir: Path,
    *,
    selection_metric: str,
    device_id: int,
    regimen: str,
) -> tuple[dict, torch.nn.Module]:
    trainer = Phase1VisualTrainer(cfg, device_id=device_id)
    best = {
        "epoch": 0,
        "score": -float("inf"),
        "state_dict": state_to_cpu(trainer.model),
        "threshold": 0.5,
        "model_source": "raw",
        "swad_n_averaged": 0,
    }
    history = []
    for epoch in range(1, cfg.phase1_epochs + 1):
        train_loss = trainer.train_epoch()
        val = eval_phase1(trainer, trainer.val_loader)
        val_threshold = float(val["best_threshold_on_this_split"]["threshold"])
        swad_active = trainer.swad.update_validation(val["loss"], step=trainer.global_step)
        trainer.scheduler.step()
        score = select_score(val, selection_metric)
        print_epoch_progress(regimen, "phase1", epoch, float(train_loss), val)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "swad_active": bool(swad_active),
            "swad_n_averaged": int(trainer.swad.n_averaged),
            "swad_state": trainer.swad.state,
            "val": val,
        }
        history.append(record)
        if score >= best["score"]:
            best = {
                "epoch": epoch,
                "score": score,
                "state_dict": state_to_cpu(trainer.model),
                "threshold": val_threshold,
                "model_source": "raw",
                "swad_n_averaged": int(trainer.swad.n_averaged),
            }
        if trainer.swad.n_averaged:
            swad_model = trainer.swad.averaged_model.module
            recompute_batchnorm(trainer.train_loader, swad_model, device=trainer.device)
            swad_val = eval_phase1(trainer, trainer.val_loader, model=swad_model)
            swad_threshold = float(swad_val["best_threshold_on_this_split"]["threshold"])
            swad_score = select_score(swad_val, selection_metric)
            record["swad_candidate"] = {"val": swad_val, "batchnorm_recomputed": True}
            print_epoch_progress(regimen, "phase1_swad", epoch, float(train_loss), swad_val)
            if swad_score >= best["score"]:
                best = {
                    "epoch": epoch,
                    "score": swad_score,
                    "state_dict": state_to_cpu(swad_model),
                    "threshold": swad_threshold,
                    "model_source": "swad",
                    "swad_n_averaged": int(trainer.swad.n_averaged),
                }
    trainer.swad.finalize()
    if trainer.swad.n_averaged:
        swad_model = trainer.swad.averaged_model.module
        recompute_batchnorm(trainer.train_loader, swad_model, device=trainer.device)
        swad_val = eval_phase1(trainer, trainer.val_loader, model=swad_model)
        swad_threshold = float(swad_val["best_threshold_on_this_split"]["threshold"])
        swad_score = select_score(swad_val, selection_metric)
        history.append(
            {
                "epoch": "final_swad",
                "swad_n_averaged": int(trainer.swad.n_averaged),
                "swad_state": trainer.swad.state,
                "val": swad_val,
                "batchnorm_recomputed": True,
            }
        )
        if swad_score >= best["score"]:
            best = {
                "epoch": cfg.phase1_epochs,
                "score": swad_score,
                "state_dict": state_to_cpu(swad_model),
                "threshold": swad_threshold,
                "model_source": "swad",
                "swad_n_averaged": int(trainer.swad.n_averaged),
            }
    trainer.model.load_state_dict(best["state_dict"])
    val_final = eval_phase1(trainer, trainer.val_loader, val_threshold=best["threshold"])
    ood_final = eval_phase1(
        trainer,
        trainer.test_loader,
        val_threshold=best["threshold"],
        include_best_threshold=False,
    )
    checkpoint = output_dir / "phase1_best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "vision_encoder_state_dict": best["state_dict"],
            "best_epoch": best["epoch"],
            "selection_metric": selection_metric,
            "selection_score": best["score"],
            "validation_threshold": best["threshold"],
            "model_source": best["model_source"],
            "swad_n_averaged": best["swad_n_averaged"],
            "config": {"source_domain_names": list(cfg.source_domain_names), "domain_column": cfg.domain_column},
        },
        checkpoint,
    )
    return {
        "best_epoch": best["epoch"],
        "selection_metric": selection_metric,
        "selection_score": best["score"],
        "validation_threshold": best["threshold"],
        "model_source": best["model_source"],
        "swad_n_averaged": best["swad_n_averaged"],
        "history": history,
        "val_best_checkpoint": val_final,
        "ood_best_checkpoint": ood_final,
        "checkpoint": str(checkpoint),
    }, trainer.model


@torch.no_grad()
def _medstyle_similarity_logits(model, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
    logits = image_emb @ text_emb.t()
    if hasattr(model, "log_temperature"):
        logits = logits / model.log_temperature.exp().clamp_min(1e-8)
    return logits


def eval_phase2_model(
    model,
    loader,
    device,
    *,
    val_threshold: float | None = None,
    include_best_threshold: bool = True,
) -> dict:
    model.eval()
    labels = []
    probs = []
    text_emb = model.encode_text([NEGATIVE_PROMPT, POSITIVE_SIMPLE])
    for images, batch_labels, *_rest in loader:
        images = images.float().to(device)
        image_emb = model.encode_image(images)
        batch_probs = F.softmax(_medstyle_similarity_logits(model, image_emb, text_emb), dim=-1)[:, 1]
        labels.extend(batch_labels.detach().cpu().long().numpy().tolist())
        probs.extend(batch_probs.detach().cpu().numpy().tolist())
    return score_summary(labels, probs, val_threshold=val_threshold, include_best_threshold=include_best_threshold)


def run_phase2(
    cfg: Config,
    output_dir: Path,
    *,
    vision_encoder: torch.nn.Module,
    selection_metric: str,
    device_id: int,
    regimen: str,
) -> tuple[dict, torch.nn.Module]:
    trainer = Phase2AlignmentTrainer(cfg, device_id=device_id, vision_encoder=vision_encoder)
    best = {"epoch": 0, "score": -float("inf"), "state_dict": state_to_cpu(trainer.model), "threshold": 0.5}
    history = []
    for epoch in range(1, cfg.medstyle_alignment_epochs + 1):
        losses = []
        for batch in trainer.train_loader:
            loss, _logits = trainer.train_step(batch)
            losses.append(loss)
        val = eval_phase2_model(trainer.model, trainer.val_loader, trainer.device)
        val_threshold = float(val["best_threshold_on_this_split"]["threshold"])
        score = select_score(val, selection_metric)
        train_loss = float(sum(losses) / max(1, len(losses)))
        print_epoch_progress(regimen, "phase2", epoch, train_loss, val)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "temperature": float(trainer.model.log_temperature.exp().detach().cpu()),
                "val": val,
            }
        )
        if score >= best["score"]:
            best = {
                "epoch": epoch,
                "score": score,
                "state_dict": state_to_cpu(trainer.model),
                "threshold": val_threshold,
            }
    trainer.model.load_state_dict(best["state_dict"])
    val_final = eval_phase2_model(trainer.model, trainer.val_loader, trainer.device, val_threshold=best["threshold"])
    ood_final = eval_phase2_model(
        trainer.model,
        trainer.test_loader,
        trainer.device,
        val_threshold=best["threshold"],
        include_best_threshold=False,
    )
    checkpoint = output_dir / "phase2_best.pt"
    torch.save(
        {
            "model_state_dict": best["state_dict"],
            "best_epoch": best["epoch"],
            "selection_metric": selection_metric,
            "selection_score": best["score"],
            "validation_threshold": best["threshold"],
            "source_domain_names": list(cfg.source_domain_names),
            "temperature_min": cfg.medstyle_temperature_min,
        },
        checkpoint,
    )
    return {
        "best_epoch": best["epoch"],
        "selection_metric": selection_metric,
        "selection_score": best["score"],
        "validation_threshold": best["threshold"],
        "history": history,
        "val_best_checkpoint": val_final,
        "ood_best_checkpoint": ood_final,
        "checkpoint": str(checkpoint),
        "trainable": trainer.model.trainable_parameter_names(),
    }, trainer.model


@torch.no_grad()
def eval_style_branch(
    trainer: Phase3StyleRemovalTrainer,
    loader,
    *,
    val_threshold: float | None = None,
    include_best_threshold: bool = True,
) -> dict:
    trainer.model.eval()
    labels = []
    probs = []
    for images, batch_labels, *_rest in loader:
        logits = trainer.image_branch_logits(images)
        batch_probs = F.softmax(logits, dim=-1)[:, 1]
        labels.extend(batch_labels.detach().cpu().long().numpy().tolist())
        probs.extend(batch_probs.detach().cpu().numpy().tolist())
    return score_summary(labels, probs, val_threshold=val_threshold, include_best_threshold=include_best_threshold)


@torch.no_grad()
def eval_ensemble(
    cfg: Config,
    medstyle_model,
    style_model,
    style_bank,
    loader,
    *,
    val_threshold: float | None = None,
    include_best_threshold: bool = True,
    weights=(0.4, 0.3, 0.3),
) -> dict:
    torch.manual_seed(cfg.seed)
    ensemble = MedStyle3DGEnsemble(medstyle_model, style_model, style_bank, weights=weights, threshold=0.5)
    labels = []
    branches = [[], [], []]
    for images, batch_labels, *_rest in loader:
        p1, p2, p3 = ensemble.branch_probabilities(images, k_styles=cfg.style_k)
        labels.extend(batch_labels.detach().cpu().long().numpy().tolist())
        for target, probs in zip(branches, (p1, p2, p3)):
            target.extend(probs.detach().cpu().numpy().tolist())
    probs = np.asarray(branches[0]) * weights[0] + np.asarray(branches[1]) * weights[1] + np.asarray(branches[2]) * weights[2]
    return {
        "weights": list(weights),
        "combined": score_summary(
            labels,
            probs,
            val_threshold=val_threshold,
            include_best_threshold=include_best_threshold,
        ),
        "branches": {
            "canonical": score_summary(labels, branches[0], include_best_threshold=include_best_threshold),
            "style_se": score_summary(labels, branches[1], include_best_threshold=include_best_threshold),
            "style_consensus": score_summary(labels, branches[2], include_best_threshold=include_best_threshold),
        },
        "labels": {"n": len(labels), "pos": int(np.sum(np.asarray(labels) == 1)), "neg": int(np.sum(np.asarray(labels) == 0))},
    }


def run_phase3(
    cfg: Config,
    output_dir: Path,
    *,
    medstyle_model: torch.nn.Module,
    selection_metric: str,
    device_id: int,
    regimen: str,
) -> tuple[dict, torch.nn.Module, object]:
    trainer = Phase3StyleRemovalTrainer(medstyle_model, cfg, source_domain_names=cfg.source_domain_names, device_id=device_id)
    best = {"epoch": 0, "score": -float("inf"), "state_dict": state_to_cpu(trainer.model), "threshold": 0.5}
    soup_members = []
    soup_size = max(1, int(cfg.phase3_soup_size))
    history = []
    for epoch in range(1, cfg.style_epochs + 1):
        losses = []
        for batch in trainer.train_loader:
            losses.append(trainer.train_step(batch))
        val = eval_style_branch(trainer, trainer.val_loader)
        val_threshold = float(val["best_threshold_on_this_split"]["threshold"])
        score = select_score(val, selection_metric)
        train_loss = float(sum(item["loss"] for item in losses) / max(1, len(losses)))
        print_epoch_progress(regimen, "phase3", epoch, train_loss, val)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_losses": losses, "val": val})
        state = state_to_cpu(trainer.model)
        insert_soup_member(
            soup_members,
            {
                "epoch": epoch,
                "score": score,
                "state_dict": state,
                "threshold": val_threshold,
                "val": val,
            },
            soup_size,
        )
        if score >= best["score"]:
            best = {
                "epoch": epoch,
                "score": score,
                "state_dict": state,
                "threshold": val_threshold,
            }
    trainer.model.load_state_dict(best["state_dict"])
    best_single_val = eval_style_branch(trainer, trainer.val_loader, val_threshold=best["threshold"])
    best_single_ood = eval_style_branch(
        trainer,
        trainer.test_loader,
        val_threshold=best["threshold"],
        include_best_threshold=False,
    )
    greedy_members = []
    greedy_attempts = []
    soup_state = None
    soup_val = None
    soup_threshold = 0.5
    soup_score = -float("inf")
    for member in soup_members:
        trial_members = [*greedy_members, member]
        trial_state = average_state_dicts([item["state_dict"] for item in trial_members])
        trainer.model.load_state_dict(trial_state)
        trial_val = eval_style_branch(trainer, trainer.val_loader)
        trial_threshold = float(trial_val["best_threshold_on_this_split"]["threshold"])
        trial_score = select_score(trial_val, selection_metric)
        accepted = not greedy_members or trial_score >= soup_score
        greedy_attempts.append(
            {
                "candidate_epoch": int(member["epoch"]),
                "candidate_score": float(member["score"]),
                "trial_size": len(trial_members),
                "trial_score": float(trial_score),
                "trial_threshold": float(trial_threshold),
                "accepted": bool(accepted),
                "val": trial_val,
            }
        )
        if accepted:
            greedy_members = trial_members
            soup_state = trial_state
            soup_val = trial_val
            soup_threshold = trial_threshold
            soup_score = trial_score
    if soup_state is None:
        soup_state = best["state_dict"]
        trainer.model.load_state_dict(soup_state)
        soup_val = eval_style_branch(trainer, trainer.val_loader)
        soup_threshold = float(soup_val["best_threshold_on_this_split"]["threshold"])
        soup_score = select_score(soup_val, selection_metric)
    else:
        trainer.model.load_state_dict(soup_state)
    val_final = eval_style_branch(trainer, trainer.val_loader, val_threshold=soup_threshold)
    ood_final = eval_style_branch(
        trainer,
        trainer.test_loader,
        val_threshold=soup_threshold,
        include_best_threshold=False,
    )
    val_ensemble_fixed = eval_ensemble(cfg, trainer.medstyle, trainer.model, trainer.style_bank, trainer.val_loader)
    fixed_threshold = float(val_ensemble_fixed["combined"]["best_threshold_on_this_split"]["threshold"])
    val_ensemble_fixed = eval_ensemble(
        cfg,
        trainer.medstyle,
        trainer.model,
        trainer.style_bank,
        trainer.val_loader,
        val_threshold=fixed_threshold,
    )
    ood_ensemble_fixed = eval_ensemble(
        cfg,
        trainer.medstyle,
        trainer.model,
        trainer.style_bank,
        trainer.test_loader,
        val_threshold=fixed_threshold,
        include_best_threshold=False,
    )
    val_branch_probs = []
    labels = []
    torch.manual_seed(cfg.seed)
    ensemble = MedStyle3DGEnsemble(trainer.medstyle, trainer.model, trainer.style_bank, weights=(0.4, 0.3, 0.3), threshold=0.5)
    for images, batch_labels, *_rest in trainer.val_loader:
        branch_probs = ensemble.branch_probabilities(images, k_styles=cfg.style_k)
        if not val_branch_probs:
            val_branch_probs = [[] for _ in branch_probs]
        for target, probs in zip(val_branch_probs, branch_probs):
            target.extend(probs.detach().cpu().numpy().tolist())
        labels.extend(batch_labels.detach().cpu().long().numpy().tolist())
    grid = grid_search_weights(labels, val_branch_probs, step=0.1)
    val_ensemble_grid = eval_ensemble(
        cfg,
        trainer.medstyle,
        trainer.model,
        trainer.style_bank,
        trainer.val_loader,
        val_threshold=float(grid["threshold"]),
        weights=tuple(grid["weights"]),
    )
    ood_ensemble_grid = eval_ensemble(
        cfg,
        trainer.medstyle,
        trainer.model,
        trainer.style_bank,
        trainer.test_loader,
        val_threshold=float(grid["threshold"]),
        include_best_threshold=False,
        weights=tuple(grid["weights"]),
    )
    soup_dir = output_dir / "phase3_soup_members"
    soup_dir.mkdir(parents=True, exist_ok=True)
    soup_checkpoints = []
    for index, member in enumerate(soup_members, start=1):
        member_path = soup_dir / f"rank_{index:02d}_epoch_{member['epoch']}.pt"
        torch.save(
            {
                "style_model_state_dict": member["state_dict"],
                "rank": index,
                "epoch": member["epoch"],
                "selection_metric": selection_metric,
                "selection_score": member["score"],
                "validation_threshold": member["threshold"],
                "val": member["val"],
            },
            member_path,
        )
        soup_checkpoints.append(str(member_path))
    checkpoint = output_dir / "phase3_best.pt"
    torch.save(
        {
            "style_model_state_dict": soup_state,
            "best_epoch": best["epoch"],
            "selection_metric": selection_metric,
            "selection_score": soup_score,
            "validation_threshold": soup_threshold,
            "model_source": "greedy_soup",
            "best_single_epoch": best["epoch"],
            "best_single_score": best["score"],
            "best_single_validation_threshold": best["threshold"],
            "soup_size_requested": soup_size,
            "soup_size_used": len(soup_members),
            "greedy_soup_size": len(greedy_members),
            "soup_members": soup_member_metadata(soup_members),
            "greedy_soup_members": soup_member_metadata(greedy_members),
            "greedy_soup_attempts": greedy_attempts,
            "soup_member_checkpoints": soup_checkpoints,
            "source_domain_names": list(trainer.style_bank.domain_names),
            "validation_grid_weights": grid,
            "style_se_reduction": cfg.style_se_reduction,
            "style_uncertainty_logit_scale": cfg.style_uncertainty_logit_scale,
            "style_arcface_scale": cfg.style_arcface_scale,
            "style_arcface_margin": cfg.style_arcface_margin,
        },
        checkpoint,
    )
    return {
        "best_epoch": best["epoch"],
        "selection_metric": selection_metric,
        "selection_score": soup_score,
        "validation_threshold": soup_threshold,
        "model_source": "greedy_soup",
        "best_single_checkpoint": {
            "epoch": best["epoch"],
            "score": best["score"],
            "validation_threshold": best["threshold"],
            "val": best_single_val,
            "ood": best_single_ood,
        },
        "soup_candidate_pool": {
            "size_requested": soup_size,
            "size_used": len(soup_members),
            "members": soup_member_metadata(soup_members),
            "member_checkpoints": soup_checkpoints,
        },
        "greedy_soup": {
            "candidate_size": len(soup_members),
            "size_used": len(greedy_members),
            "members": soup_member_metadata(greedy_members),
            "attempts": greedy_attempts,
            "score": soup_score,
            "validation_threshold": soup_threshold,
        },
        "history": history,
        "val_best_checkpoint": val_final,
        "ood_best_checkpoint": ood_final,
        "source_domain_names": list(trainer.style_bank.domain_names),
        "style_se_reduction": cfg.style_se_reduction,
        "style_uncertainty_logit_scale": cfg.style_uncertainty_logit_scale,
        "ensemble_fixed_article_weights": {
            "validation_threshold": fixed_threshold,
            "val": val_ensemble_fixed,
            "ood": ood_ensemble_fixed,
        },
        "ensemble_validation_grid_weights": {
            "grid": grid,
            "val": val_ensemble_grid,
            "ood": ood_ensemble_grid,
        },
        "checkpoint": str(checkpoint),
        "source_domain_names": list(trainer.style_bank.domain_names),
    }, trainer.model, trainer.style_bank


def run_regimen(args, split_dir: Path, output_root: Path, regimen: str, domain_column: str) -> dict:
    train_csv = split_dir / ("TrainPlusTestID.csv" if regimen == "train_plus_test_id" else "Train.csv")
    cfg = make_config(args, split_dir, output_root / regimen, train_csv, domain_column)
    regimen_dir = output_root / regimen
    regimen_dir.mkdir(parents=True, exist_ok=True)
    data_loaders = make_data_loaders(cfg)
    prompted_loaders = make_prompted_data_loaders(cfg)
    data_summary = {
        "train_rows": len(data_loaders[0].dataset),
        "valid_rows": len(data_loaders[1].dataset),
        "ood_rows": len(data_loaders[2].dataset),
        "prompted_train_rows": len(prompted_loaders[0].dataset),
        "source_domain_names": list(cfg.source_domain_names),
        "domain_column": cfg.domain_column,
        "domain_aware_batching": bool(cfg.domain_aware_batching),
        "mixstyle": bool(cfg.use_mixstyle),
    }
    print(f"[{regimen}] data {json.dumps(data_summary)}", flush=True)
    phase1, vision_encoder = run_phase1(
        cfg,
        regimen_dir,
        selection_metric=args.selection_metric,
        device_id=args.device_id,
        regimen=regimen,
    )
    print(f"[{regimen}] phase1 best_epoch={phase1['best_epoch']} val_auc={phase1['val_best_checkpoint']['auroc']:.4f} ood_auc={phase1['ood_best_checkpoint']['auroc']:.4f}", flush=True)
    phase2, medstyle_model = run_phase2(
        cfg,
        regimen_dir,
        vision_encoder=vision_encoder,
        selection_metric=args.selection_metric,
        device_id=args.device_id,
        regimen=regimen,
    )
    print(f"[{regimen}] phase2 best_epoch={phase2['best_epoch']} val_auc={phase2['val_best_checkpoint']['auroc']:.4f} ood_auc={phase2['ood_best_checkpoint']['auroc']:.4f}", flush=True)
    phase3, _style_model, _style_bank = run_phase3(
        cfg,
        regimen_dir,
        medstyle_model=medstyle_model,
        selection_metric=args.selection_metric,
        device_id=args.device_id,
        regimen=regimen,
    )
    print(f"[{regimen}] phase3 best_epoch={phase3['best_epoch']} val_auc={phase3['val_best_checkpoint']['auroc']:.4f} ood_auc={phase3['ood_best_checkpoint']['auroc']:.4f}", flush=True)
    return {"data": data_summary, "phase1": phase1, "phase2": phase2, "phase3": phase3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the article split protocol with validation-selected checkpoints.")
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--patch-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/tmp/generalization_article_protocol"))
    parser.add_argument("--text-model", type=Path)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--domain-column")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--patch-column", default="patch_name")
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--phase1-epochs", type=int, default=300)
    parser.add_argument("--phase1-batch-size", type=int, default=128)
    parser.add_argument("--phase1-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--phase2-epochs", type=int, default=100)
    parser.add_argument("--phase2-batch-size", type=int, default=128)
    parser.add_argument("--phase2-lr", type=float, default=1e-4)
    parser.add_argument("--phase3-epochs", type=int, default=100)
    parser.add_argument("--phase3-batch-size", type=int, default=128)
    parser.add_argument("--phase3-lr", type=float, default=0.008)
    parser.add_argument("--phase3-soup-size", type=int, default=5)
    parser.add_argument("--style-k", type=int, default=20)
    parser.add_argument(
        "--selection-metric",
        choices=("auroc", "average_precision", "f1", "fixed_f1", "f1_at_0_5", "best_f1", "threshold_f1"),
        default="best_f1",
    )
    parser.add_argument("--skip-preflight", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed, deterministic=args.deterministic)
    started = time.time()
    report = read_report(args.split_dir)
    domain_column = args.domain_column or report.get("style_column") or report.get("split_column") or "manufacturer"
    output_root = args.output_root / time.strftime("article_protocol_%Y%m%d_%H%M%S")
    output_root.mkdir(parents=True, exist_ok=True)
    split_dir = args.split_dir
    preflight = preflight_patch_root(split_dir, args.patch_root, patch_column=args.patch_column)
    if not args.skip_preflight and not preflight["ok"]:
        out = {"split_dir": str(split_dir), "preflight": preflight, "error": "Missing patch files for generated CSVs."}
        (output_root / "protocol_results.json").write_text(json.dumps(out, indent=2, sort_keys=True))
        raise SystemExit(f"Patch preflight failed; see {output_root / 'protocol_results.json'}")

    results = {
        "seed": int(args.seed),
        "deterministic": bool(args.deterministic),
        "split_dir": str(split_dir),
        "patch_root": str(args.patch_root),
        "domain_column": domain_column,
        "preflight": preflight,
        "regimens": {},
    }
    for regimen in ("train", "train_plus_test_id"):
        results["regimens"][regimen] = run_regimen(args, split_dir, output_root, regimen, domain_column)
        (output_root / "protocol_results.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    results["elapsed_sec"] = time.time() - started
    out_path = output_root / "protocol_results.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"ARTICLE_SPLIT_PROTOCOL_OK {out_path} elapsed={results['elapsed_sec']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
