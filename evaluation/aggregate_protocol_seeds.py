from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path


def numeric(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        if math.isfinite(value):
            return value
    return None


def add_metric(out: dict[str, float], prefix: str, metrics: dict, include_best: bool) -> None:
    for key in ("auroc", "average_precision", "prob_mean", "prob_std", "pos_neg_mean_gap"):
        value = numeric(metrics.get(key))
        if value is not None:
            out[f"{prefix}.{key}"] = value
    for group_name, group in (
        ("fixed_0_5", metrics.get("fixed_0_5", {})),
        ("at_validation_threshold", metrics.get("at_validation_threshold", {})),
    ):
        if isinstance(group, dict):
            for key in ("f1", "precision", "recall", "specificity", "balanced_accuracy", "threshold"):
                value = numeric(group.get(key))
                if value is not None:
                    out[f"{prefix}.{group_name}.{key}"] = value
    if include_best and isinstance(metrics.get("best_threshold_on_this_split"), dict):
        group = metrics["best_threshold_on_this_split"]
        for key in ("f1", "precision", "recall", "threshold"):
            value = numeric(group.get(key))
            if value is not None:
                out[f"{prefix}.best_threshold.{key}"] = value


def flatten_result(protocol: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for regimen, regimen_data in protocol.get("regimens", {}).items():
        for phase in ("phase1", "phase2", "phase3"):
            phase_data = regimen_data.get(phase)
            if not isinstance(phase_data, dict):
                continue
            for key in ("best_epoch", "selection_score", "validation_threshold", "swad_n_averaged"):
                value = numeric(phase_data.get(key))
                if value is not None:
                    out[f"{regimen}.{phase}.{key}"] = value
            if isinstance(phase_data.get("val_best_checkpoint"), dict):
                add_metric(
                    out,
                    f"{regimen}.{phase}.val",
                    phase_data["val_best_checkpoint"],
                    include_best=True,
                )
            if isinstance(phase_data.get("ood_best_checkpoint"), dict):
                add_metric(
                    out,
                    f"{regimen}.{phase}.ood",
                    phase_data["ood_best_checkpoint"],
                    include_best=False,
                )
            if phase == "phase3":
                for ensemble_name, source_key in (
                    ("ensemble_fixed", "ensemble_fixed_article_weights"),
                    ("ensemble_grid", "ensemble_validation_grid_weights"),
                ):
                    ensemble = phase_data.get(source_key)
                    if not isinstance(ensemble, dict):
                        continue
                    if isinstance(ensemble.get("val", {}).get("combined"), dict):
                        add_metric(
                            out,
                            f"{regimen}.{phase}.{ensemble_name}.val",
                            ensemble["val"]["combined"],
                            include_best=True,
                        )
                    if isinstance(ensemble.get("ood", {}).get("combined"), dict):
                        add_metric(
                            out,
                            f"{regimen}.{phase}.{ensemble_name}.ood",
                            ensemble["ood"]["combined"],
                            include_best=False,
                        )
    return out


def infer_seed(path: Path, protocol: dict, fallback: str) -> str:
    value = protocol.get("seed")
    if value is not None:
        return str(value)
    text = str(path)
    match = re.search(r"seed[_=-]?(\d+)", text)
    if match:
        return match.group(1)
    for parent in [path.parent, *path.parents[:5]]:
        command = parent / "command.sh"
        if command.exists():
            match = re.search(r"--seed\s+(\d+)", command.read_text())
            if match:
                return match.group(1)
    return fallback


def collect_paths(args) -> list[Path]:
    paths = [Path(item) for item in args.results]
    for root in args.root:
        paths.extend(Path(root).glob("**/protocol_results.json"))
    unique = sorted({path.resolve() for path in paths if path.exists()})
    return unique


def aggregate(per_seed: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = sorted({key for metrics in per_seed.values() for key in metrics})
    out = {}
    for key in keys:
        values = [metrics[key] for metrics in per_seed.values() if key in metrics]
        if not values:
            continue
        out[key] = {
            "n": len(values),
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="*")
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path)
    args = parser.parse_args()

    per_seed = {}
    inputs = {}
    for idx, path in enumerate(collect_paths(args), start=1):
        protocol = json.loads(path.read_text())
        seed = infer_seed(path, protocol, f"result_{idx}")
        per_seed[seed] = flatten_result(protocol)
        inputs[seed] = str(path)

    summary = aggregate(per_seed)
    payload = {"inputs": inputs, "per_seed": per_seed, "aggregate": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))

    if args.csv_output:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_output.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric", "n", "mean", "std", "min", "max"])
            for key, values in summary.items():
                writer.writerow([key, values["n"], values["mean"], values["std"], values["min"], values["max"]])

    print(f"AGGREGATED_SEED_METRICS {args.output}")
    if args.csv_output:
        print(f"AGGREGATED_SEED_METRICS_CSV {args.csv_output}")


if __name__ == "__main__":
    main()
