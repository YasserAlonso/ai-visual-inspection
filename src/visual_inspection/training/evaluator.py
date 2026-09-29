"""Ultralytics evaluation and normalization of its official metric implementation."""

from pathlib import Path
from datetime import datetime, timezone


def serialize_metrics(metrics, names: dict[int, str] | None = None) -> tuple[dict, dict]:
    if isinstance(metrics, dict):
        keys = {
            "precision": "metrics/precision(B)",
            "recall": "metrics/recall(B)",
            "map50": "metrics/mAP50(B)",
            "map50_95": "metrics/mAP50-95(B)",
        }
        missing = [key for key in keys.values() if key not in metrics or metrics[key] is None]
        if missing:
            raise ValueError(f"Ultralytics metrics are missing: {missing}")
        return {name: float(metrics[key]) for name, key in keys.items()}, {}
    box = getattr(metrics, "box", None)
    if box is None:
        raise ValueError("Ultralytics did not return detection box metrics")
    keys = {"precision": "mp", "recall": "mr", "map50": "map50", "map50_95": "map"}
    normalized = {}
    for output, attribute in keys.items():
        value = getattr(box, attribute, None)
        if value is None:
            raise ValueError(f"Ultralytics metric is missing: {attribute}")
        normalized[output] = float(value)
    names = names or getattr(metrics, "names", {})
    per_class = {}
    class_ids = list(getattr(box, "ap_class_index", []))
    if hasattr(metrics, "class_result") and class_ids:
        for index, class_id in enumerate(class_ids):
            precision, recall, ap50, ap50_95 = metrics.class_result(index)
            per_class[str(names.get(int(class_id), int(class_id)))] = {
                "precision": float(precision),
                "recall": float(recall),
                "ap50": float(ap50),
                "ap50_95": float(ap50_95),
            }
    else:
        class_ap = getattr(box, "maps", [])
        per_class = {
            str(names.get(i, i)): {"ap50_95": float(value)}
            for i, value in enumerate(class_ap)
        }
    return normalized, per_class


def evaluate(
    model_path: Path,
    dataset_yaml: Path,
    *,
    split: str = "test",
    device="auto",
    output: Path = Path("artifacts/evaluations"),
):
    from ultralytics import YOLO
    from visual_inspection.training.environment import environment_metadata, select_device

    hardware = environment_metadata()
    selected = select_device(device, hardware)
    framework_device = None if selected == "auto" else selected
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    run_name = f"{model_path.stem}_{split}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S.%fZ}"
    result = model.val(
        data=str(dataset_yaml),
        split=split,
        device=framework_device,
        plots=True,
        project=str(output),
        name=run_name,
        exist_ok=False,
    )
    metrics, per_class = serialize_metrics(result)
    return metrics, per_class, result
