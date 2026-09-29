## [Unreleased] - v0.2.0 AeroInspect AI Domain Pivot

- Reframed project documentation as AeroInspect AI, an AI-assisted aerospace inspection and maintenance platform.
- Preserved Phase 0 audit and Phase 1 YOLO detection infrastructure; generalized training examples so dataset configuration supplies task classes.
- Selected Aircraft_Fuselage_DET2023 as the first real object-detection benchmark, pending manual source acquisition and package verification.
- Added dataset provenance/access documentation, ignored source-data location, canonical-class configuration placeholder, and aerospace YOLO11n baseline config.
- Updated the README with the fuselage-inspection milestone and future module roadmap. No source data was accessed and no real-data conversion, audit, training, evaluation, release, or version tag has been performed.
- Retained the prior CUDA synthetic smoke result strictly as software validation, with no inspection-quality claim.

## [0.1.0] - Initial Dataset Audit Pipeline

- Added installable Python 3.11+ package, CLI, YAML configuration, and development setup.
- Added image validation, recursive scanning, SHA-256 and 64-bit pHash fingerprints.
- Added BK-tree near-duplicate search, duplicate grouping, and cross-split leakage detection.
- Added pandas metadata, statistical distributions, health rules, and CSV/JSON/Markdown reports.
- Added synthetic fixtures, regression tests, and reproducible demonstration generator.
- Documented Phase 0 limits and the roadmap; no model training is implemented.
- Verified v0.1.0 installation, CLI entry points, report contents, and health boundaries.
- Added same-split demo duplicates and end-to-end regression checks for every report.
- Rejected case-insensitive split aliases and narrowed image validation exception handling.
