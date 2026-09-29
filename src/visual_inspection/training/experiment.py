"""Per-run metadata persistence, kept separate from ignored model artifacts."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess

import yaml


def create_experiment(
    root: Path, name: str, config: dict, environment: dict, audit_summary: dict
) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip(".-")
    if not safe_name:
        raise ValueError("experiment name must contain letters or numbers")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = root / f"{safe_name}_{timestamp}"
    path.mkdir(parents=True, exist_ok=False)
    (path / "plots").mkdir()
    (path / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        git_status = subprocess.run(
            ["git", "status", "--short"], capture_output=True, text=True, check=True
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        git_commit, git_status = None, ["unavailable"]
    environment = {**environment, "git_commit": git_commit, "git_status_short": git_status}
    (path / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    (path / "audit_summary.json").write_text(json.dumps(audit_summary, indent=2), encoding="utf-8")
    created = datetime.now(timezone.utc).isoformat()
    (path / "experiment.json").write_text(
        json.dumps(
            {
                "experiment_name": name,
                "created_at_utc": created,
                "model": config.get("model"),
                "dataset_path": config.get("dataset"),
                "audit_health": audit_summary.get("health"),
                "seed": config.get("seed"),
                "training_config": config,
                "environment": environment,
                "dataset_warning": (
                    "Baseline trained on utsv3_clean_final2 with documented label noise: "
                    "60 tiny annotations remain QUESTIONABLE after manual adjudication; "
                    "0 reviewed tiny annotations were CLEARLY_INVALID."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def write_metrics(
    path: Path,
    metrics: dict,
    per_class: dict | None = None,
    run_label: str = "BASELINE TRAINING",
) -> None:
    payload = {"run_label": run_label, "metrics": metrics, "per_class": per_class or {}}
    (path / "metrics.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )


def copy_training_plots(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for plot in source.glob("*.png"):
        shutil.copy2(plot, destination / plot.name)


def write_training_completion(
    path: Path, *, started_at: str, duration_seconds: float, epochs_completed: int,
    configured_epochs: int, patience: int, best_epoch: int | None,
    best_map50: float | None, best_map50_95: float | None,
) -> None:
    payload = {
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_duration_seconds": duration_seconds,
        "epochs_completed": epochs_completed,
        "configured_epochs": configured_epochs,
        "early_stopping_triggered": epochs_completed < configured_epochs,
        "patience": patience,
        "best_epoch": best_epoch,
        "best_validation_map50": best_map50,
        "best_validation_map50_95": best_map50_95,
    }
    (path / "training_completion.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
