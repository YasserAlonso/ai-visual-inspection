# AeroInspect AI

**AI-Assisted Aerospace Inspection & Maintenance Platform**

A Python foundation for auditing image datasets before model training. Phase 0 detects invalid images, exact duplicates, perceptual matches, and train/validation/test leakage, and produces inspectable reports without modifying source data. Phase 1 provides reusable YOLO object-detection training, evaluation, annotation audit, and experiment tracking infrastructure.

AeroInspect AI is intended to highlight potential defects or regions of interest for qualified human review. It is not an autonomous airworthiness decision system, a safety certification system, or a replacement for certified inspectors.

The first real aerospace benchmark is **Aircraft_Fuselage_DET2023** for fuselage defect detection. Its official record is not accessible from the current environment and the dataset is not present locally, so no real-data results are claimed. See the [dataset record and acquisition status](docs/aircraft_fuselage_det2023.md). The earlier [dataset feasibility study](docs/aerospace_dataset_selection.md) is retained as historical research.

Current release: **v0.1.0 — Dataset Audit Pipeline**. The v0.2.0 development work remains unreleased and untagged.

## Implemented capabilities

- Recursive image validation, SHA-256 exact duplicate detection, perceptual near-duplicate grouping, and cross-split leakage reporting.
- YOLO detection annotation validation, class and box statistics, audit gating, and configurable training.
- PyTorch/CUDA environment reporting, reproducible experiment metadata, and Ultralytics evaluation metrics.
- A synthetic one-epoch smoke test that verifies software execution only; it does not measure aerospace inspection quality.

## Dataset and model status

The selected first task is **aircraft fuselage defect detection**. Intended semantic categories are corrosion, cracks, dents, and missing/damaged fasteners, but the source's authoritative names and mapping are pending inspection of the downloaded package. The source archive belongs in `data/external/aircraft_fuselage_det2023/original/`; it remains ignored and immutable. Processed labels and split copies belong in `data/processed/aircraft_fuselage_det2023/`.

The official IEEE DataPort record is inaccessible from this environment and its license, annotation package, exact counts, classes, and split remain unverified. No dataset was downloaded. Complete the manual acquisition step in [the dataset record](docs/aircraft_fuselage_det2023.md) before conversion, audit, splitting, or real training. The aerospace baseline configuration is [aircraft_fuselage_baseline.yaml](configs/training/aircraft_fuselage_baseline.yaml); do not run it until real data passes the preparation and audit gates.

## Installation

Python 3.11 or newer is required. From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

For Phase 1 detection support, install a PyTorch build suitable for your operating system, GPU, and CUDA runtime, then install the training extra:

```powershell
python -m pip install torch torchvision
python -m pip install -e ".[dev,training]"
```

For the current Windows/NVIDIA setup, see the [official PyTorch install selector](https://pytorch.org/get-started/locally/) for the matching CUDA wheel. A CUDA wheel alone does not guarantee that a compatible GPU driver is available.

## Phase 0 dataset audit

The Phase 0 CLI remains domain-neutral and does not modify input data. Example:

```powershell
python -m visual_inspection.cli audit --dataset C:/datasets/source_copy --output data/reports/aebad
```

To create an audit demonstration dataset:

```powershell
python scripts/create_demo_dataset.py --output data/raw/demo
python scripts/audit_dataset.py --dataset data/raw/demo --output data/reports/demo
```

Reports must be outside the dataset directory. Inputs should remain unchanged during a run. See the CLI help and `configs/dataset_audit.example.yaml` for available settings.

## Phase 1 detection infrastructure

The existing baseline uses the pretrained Ultralytics YOLO11n detection checkpoint as one practical small detector; the architecture is not a project-wide commitment. Detection datasets must use YOLO labels and a dataset YAML. Start from `configs/training/dataset.example.yaml`, replacing the placeholder class with the exact classes in the dataset. The example is deliberately domain-neutral.

The trainer validates images, annotations, classes, boxes, and dataset leakage before training. It records environment, config, audit results, metrics, and checkpoint references under `artifacts/experiments/`. Evaluation supports validation or held-out test splits and reports the metrics returned by Ultralytics: precision, recall, AP50/mAP50, and mAP50-95.

The pipeline expects object-detection boxes in YOLO format. Do not infer boxes or class semantics from prose; confirm the source package's native annotation format and mapping first.

## Synthetic software smoke test

A tiny random synthetic dataset can check that the object-detection software stack starts and writes artifacts:

```powershell
python scripts/create_smoke_dataset.py --output artifacts/smoke/dataset
python scripts/train_baseline.py --config artifacts/smoke/training-smoke.yaml
```

Label every result **SOFTWARE SMOKE TEST ONLY**. A prior one-epoch CUDA smoke run completed on an RTX 5070 Ti Laptop GPU. Validation precision/recall/AP50/mAP50-95 was 0.0000; one synthetic test pass produced 0.0123/0.5000/0.0354/0.0212. These results describe eight synthetic images and are not aerospace quality measurements. Outputs are ignored under `artifacts/smoke_release/`.

## Next milestone

1. Acquire Aircraft_Fuselage_DET2023 from the [official IEEE DataPort record](https://ieee-dataport.org/documents/aircraft-fuselagedet2023-aircraft-fuselage-defect-detection-dataset) and preserve the untouched source with checksums and terms.
2. Verify the class map, native boxes, source counts, image sizes, and any official split. Convert reproducibly only if needed.
3. Audit annotations and images, check exact/perceptual duplicates and capture groups, then preserve official splits or create deterministic leakage-aware train/validation/test manifests.
4. Train one YOLO11n transfer-learning baseline only after the dataset passes audit; report validation separately from the one-time held-out test evaluation.

## First real aerospace model

The first planned model is an object detector for Aircraft_Fuselage_DET2023. It will identify source-verified defect classes with bounding boxes, using pretrained YOLO11n transfer learning. The split will preserve an official source split where available; otherwise it will use capture-aware grouping and deterministic manifests, with duplicate and leakage checks before training. Phase 0 image/annotation audit gates training. Validation metrics guide the baseline, and held-out test metrics are reported once after configuration is frozen. Dataset access and terms, class mapping, annotations, split, and all real-data metrics remain pending; no baseline has been trained. This system is AI-assisted defect detection for human inspection review, not a replacement for certified aircraft inspectors.

Roadmap: Module 1, aircraft fuselage defect detection; future Module 2, engine blade anomaly inspection; Module 3, foreign object debris detection; Module 4, aerospace manufacturing defect inspection; Module 5, predictive maintenance from sensor/time-series data; Module 6, multimodal inspection assistant; Module 7, maintenance-document RAG; Module 8, automated inspection report generation. These future modules are not being built now.

For the dataset comparison, license/access assessment, and complete preparation strategy, see [the aerospace dataset selection report](docs/aerospace_dataset_selection.md).
