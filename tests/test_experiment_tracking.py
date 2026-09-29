import json

from visual_inspection.training.environment import select_device
from visual_inspection.training.evaluator import serialize_metrics
from visual_inspection.training.experiment import create_experiment, write_metrics


def test_experiment_creation_and_metrics(tmp_path):
    path = create_experiment(
        tmp_path, "road damage", {"seed": 42}, {"python_version": "3.12"}, {"health": "WARNING"}
    )
    assert path.is_dir()
    assert (path / "plots").is_dir()
    assert json.loads((path / "environment.json").read_text())["python_version"] == "3.12"
    experiment = json.loads((path / "experiment.json").read_text())
    assert experiment["experiment_name"] == "road damage"
    assert experiment["audit_health"] == "WARNING"
    assert experiment["seed"] == 42
    write_metrics(
        path, {"precision": 0.5, "map50": 0.4}, {"crack": 0.3}, "SOFTWARE SMOKE TEST ONLY"
    )
    saved = json.loads((path / "metrics.json").read_text())
    assert saved["per_class"]["crack"] == 0.3
    assert saved["run_label"] == "SOFTWARE SMOKE TEST ONLY"


def test_metric_serialization_uses_framework_values():
    metrics, per_class = serialize_metrics(
        {
            "metrics/precision(B)": 0.8,
            "metrics/recall(B)": 0.7,
            "metrics/mAP50(B)": 0.75,
            "metrics/mAP50-95(B)": 0.52,
        }
    )
    assert metrics == {"precision": 0.8, "recall": 0.7, "map50": 0.75, "map50_95": 0.52}
    assert per_class == {}


def test_metric_serialization_includes_per_class_detection_metrics():
    class Box:
        mp, mr, map50, map = 0.6, 0.5, 0.4, 0.3
        ap_class_index = [0, 2]

    class Metrics:
        box = Box()

        @staticmethod
        def class_result(index):
            return [(0.7, 0.6, 0.5, 0.4), (0.5, 0.4, 0.3, 0.2)][index]

    overall, classes = serialize_metrics(Metrics(), {0: "Dent", 1: "Fastener Damage", 2: "Rupture"})
    assert overall == {"precision": 0.6, "recall": 0.5, "map50": 0.4, "map50_95": 0.3}
    assert classes["Dent"] == {"precision": 0.7, "recall": 0.6, "ap50": 0.5, "ap50_95": 0.4}
    assert classes["Rupture"]["ap50"] == 0.3


def test_device_fallback_and_explicit_device():
    assert select_device("auto", {"cuda_available": False}) == "cpu"
    assert select_device("auto", {"cuda_available": True}) == "auto"
    assert select_device("cpu", {"cuda_available": True}) == "cpu"
