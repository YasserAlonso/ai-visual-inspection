"""Baseline fine-tuning through Ultralytics' supported YOLO training API."""

from pathlib import Path
import random
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import time
import numpy as np

from visual_inspection.training.dataset import load_dataset_yaml, write_absolute_dataset_yaml
from visual_inspection.training.environment import environment_metadata, select_device
from visual_inspection.training.experiment import (
    copy_training_plots,
    create_experiment,
    write_metrics,
    write_training_completion,
)


def run_training(config, *, allow_failed_audit: bool = False) -> Path:
    from ultralytics import YOLO
    from visual_inspection.audit import audit_dataset
    from visual_inspection.config import AuditConfig
    import torch

    dataset = load_dataset_yaml(config.dataset)
    for split, image_dir in dataset.splits.items():
        if image_dir.parent.parent.resolve() != dataset.root:
            raise ValueError(f"{split} images must follow <dataset>/<split>/images layout")
    names = {key: value for key, value in zip(("train", "val", "test"), dataset.splits)}
    audit_config = AuditConfig(
        train_name=names["train"],
        validation_name=names["val"],
        test_name=names.get("test", "test"),
        phash_distance_threshold=config.phash_distance_threshold,
        corrupted_image_failure_threshold=config.corrupted_image_failure_threshold,
    )
    audit = audit_dataset(
        dataset.root,
        config.audit_output_directory,
        audit_config,
        yolo_dataset_yaml=config.dataset,
        minimum_box_area=config.minimum_box_area,
        imbalance_warning_ratio=config.imbalance_warning_ratio,
    )
    if audit.summary["health"] == "FAIL" and not (allow_failed_audit or config.allow_failed_audit):
        raise RuntimeError(f"Training blocked: audit health FAIL. Review {audit.report_directory}")

    hardware = environment_metadata()
    selected_device = select_device(config.device, hardware)
    print(f"PyTorch: {hardware['pytorch_version']}")
    print(f"CUDA available: {hardware['cuda_available']}")
    print(f"CUDA version: {hardware['cuda_version']}")
    print(f"GPU: {hardware['gpu_name'] or 'none'}")
    print(
        "Selected device: "
        f"{'CUDA (framework automatic selection)' if selected_device == 'auto' else selected_device}"
    )
    framework_device = None if selected_device == "auto" else selected_device
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False

    saved_config = asdict(config)
    saved_config = {
        key: str(value) if isinstance(value, Path) else value for key, value in saved_config.items()
    }
    experiment = create_experiment(
        config.project_directory,
        config.experiment_name,
        saved_config,
        hardware | {"ultralytics_version": __import__("ultralytics").__version__},
        audit.summary,
    )
    dataset_file = write_absolute_dataset_yaml(dataset, experiment / "dataset.yaml")
    model = YOLO(config.model)
    training_started_at = datetime.now(timezone.utc).isoformat()
    training_started = time.perf_counter()
    model.train(
        data=str(dataset_file),
        epochs=config.epochs,
        imgsz=config.image_size,
        batch=config.batch_size,
        device=framework_device,
        workers=config.workers,
        seed=config.seed,
        deterministic=True,
        patience=config.patience,
        project=str(experiment),
        name="training",
        exist_ok=False,
        plots=True,
        save=True,
        val=True,
        verbose=True,
        **config.augmentation,
    )
    training_duration = time.perf_counter() - training_started
    trainer = model.trainer
    save_dir = Path(trainer.save_dir)
    best = Path(getattr(trainer, "best", save_dir / "weights" / "best.pt"))
    if not best.exists():
        raise RuntimeError("Training completed but the best checkpoint was not found")
    metrics_result = YOLO(str(best))
    validation = metrics_result.val(
        data=str(dataset_file),
        split="val",
        device=framework_device,
        plots=True,
        project=str(experiment),
        name="best_validation",
        exist_ok=False,
    )
    from visual_inspection.training.evaluator import serialize_metrics
    from visual_inspection.training.error_analysis import generate_validation_error_examples

    metrics, per_class = serialize_metrics(validation, dataset.names)
    write_metrics(experiment, metrics, per_class, config.run_label)
    error_counts = generate_validation_error_examples(
        metrics_result, dataset, experiment / "validation_error_analysis", device="0"
    )
    (experiment / "validation_error_analysis" / "counts.json").write_text(
        __import__("json").dumps(error_counts, indent=2), encoding="utf-8"
    )
    results_file = save_dir / "results.csv"
    epoch_rows = list(csv.DictReader(results_file.open(encoding="utf-8-sig", newline="")))
    map_column = "metrics/mAP50(B)"
    best_row = max(epoch_rows, key=lambda row: float(row[map_column])) if epoch_rows else None
    best_epoch = int(float(best_row["epoch"])) + 1 if best_row else None
    last_losses = {}
    best_losses = {}
    if epoch_rows:
        last = epoch_rows[-1]
        loss_columns = (
            ("train_box", "train/box_loss"),
            ("train_classification", "train/cls_loss"),
            ("train_dfl", "train/dfl_loss"),
            ("validation_box", "val/box_loss"),
            ("validation_classification", "val/cls_loss"),
            ("validation_dfl", "val/dfl_loss"),
        )
        for key, column in loss_columns:
            if column in last:
                last_losses[key] = float(last[column])
        if best_row:
            for key, column in loss_columns:
                if column in best_row:
                    best_losses[key] = float(best_row[column])
    write_training_completion(
        experiment,
        started_at=training_started_at,
        duration_seconds=training_duration,
        epochs_completed=len(epoch_rows),
        configured_epochs=config.epochs,
        patience=config.patience,
        best_epoch=best_epoch,
        best_map50=float(best_row[map_column]) if best_row else None,
        best_map50_95=float(best_row["metrics/mAP50-95(B)"]) if best_row else None,
    )
    copy_training_plots(save_dir, experiment / "plots" / "training")
    copy_training_plots(experiment / "best_validation", experiment / "plots" / "best_validation")
    (experiment / "model_reference.txt").write_text(str(best.resolve()), encoding="utf-8")
    (experiment / "training_summary.md").write_text(
        "# Baseline training summary\n\n"
        f"**{config.run_label}. Synthetic results do not indicate inspection quality.**\n\n"
        f"- Model: `{config.model}`\n- Dataset YAML: `{config.dataset}`\n"
        f"- Epochs configured/completed: {config.epochs}/{len(epoch_rows)}\n- Selected device: {selected_device}\n"
        f"- Dataset audit: {audit.summary['health']}\n"
        f"- Best epoch by validation mAP50: {best_epoch}\n"
        f"- Final training losses: {last_losses}\n"
        f"- Best-epoch losses: {best_losses}\n"
        f"- Precision: {metrics['precision']:.4f}\n- Recall: {metrics['recall']:.4f}\n"
        f"- AP50 / mAP50: {metrics['map50']:.4f}\n"
        f"- mAP50-95: {metrics['map50_95']:.4f}\n"
        f"- Per-class validation metrics: {per_class}\n"
        f"- Validation error examples: {error_counts}\n"
        "\nMetrics use Ultralytics' validator. Inspect results.csv and plots for train/validation curves.\n",
        encoding="utf-8",
    )
    return experiment
