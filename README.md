# AI Visual Inspection Platform

A Python foundation for auditing image datasets before model training. Phase 0 detects
invalid images, byte-identical duplicates, perceptual matches, and train/validation/test
leakage, and produces inspectable reports without modifying the dataset.

The long-term vision is an end-to-end computer vision platform for component and defect
detection, image-and-text questions, technical-document retrieval, and inspection reports,
with APIs, a web interface, experiment tracking, and production model monitoring.
**Only Phase 0 is implemented.** No heavyweight ML training frameworks are required.

Current release: **v0.1.0 — Dataset Audit Pipeline**.

## Installation

Python 3.11 or newer is required. From the repository root, in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
```

On macOS/Linux, use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`.
Install the package before running script wrappers or module commands. Editable installation
(`pip install -e .`, or `pip install -e ".[dev]"` for development) is recommended when working
from this repository; a regular `pip install .` also works.

Check the CLI from the project root:

```powershell
.\.venv\Scripts\python.exe -m visual_inspection.cli --help
```

## Repository structure

```text
ai_visual_inspection/
├── README.md
├── CHANGELOG.md
├── pyproject.toml
├── .gitignore
├── .env.example
├── configs/dataset_audit.example.yaml
├── data/{raw,processed,reports}/.gitkeep
├── src/visual_inspection/
│   ├── __init__.py
│   ├── audit.py
│   ├── cli.py
│   ├── config.py
│   └── data/
│       ├── __init__.py
│       ├── scanner.py
│       ├── hashing.py
│       ├── duplicates.py
│       ├── leakage.py
│       ├── validation.py
│       ├── statistics.py
│       └── report.py
├── tests/
│   ├── conftest.py
│   ├── test_hashing.py
│   ├── test_duplicates.py
│   ├── test_leakage.py
│   ├── test_validation.py
│   ├── test_pipeline.py
│   └── test_release.py
└── scripts/
    ├── audit_dataset.py
    └── create_demo_dataset.py
```

Start with `audit.py` for orchestration, `config.py` for settings, `scanner.py` and
`validation.py` for ingestion, `duplicates.py` for matching, and `report.py` for health policy.
Library users can call `audit_dataset(Path(...), Path(...), AuditConfig(...))`; the result
contains images, exact/near pairs, leakage DataFrames, summary, and report directory.

## Dataset layout

```text
dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Images are scanned recursively within each images directory, with case-insensitive
`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tif`, and `.tiff` extensions. Files with unsupported
extensions are recorded as invalid. Missing image or label directories are reported.
Unrecognized root directories are reported and not scanned. Root-level files and labels
are not image inputs. Split names are configurable. An absent test split triggers a warning.
Symbolic-link files are invalid; symbolic-link directories and enumeration failures abort
the audit to avoid silently reporting on an incomplete dataset.

YOLO annotation contents, missing per-image labels, class counts, box validity, and object
sizes are **not evaluated yet**. `validation.py` is the extension point for annotation checks.

## Run an audit

```powershell
.\.venv\Scripts\python.exe -m visual_inspection.cli audit --dataset ./dataset --output ./data/reports --phash-threshold 5
.\.venv\Scripts\python.exe scripts/audit_dataset.py --dataset ./dataset --config configs/dataset_audit.example.yaml
```

After activating the environment, `visual-inspection audit ...` is also available.
Explicit CLI values override YAML settings; omitted values preserve YAML or defaults.
Use `--train-name`, `--validation-name`, `--test-name`, and
`--corrupted-image-failure-threshold` to override the other exposed settings.
Unknown YAML keys and invalid settings are rejected. No environment variables are required.
Reports must be outside the dataset directory. Inputs should remain unchanged during a run.

To reproduce the demo (the generator refuses to overwrite an existing destination):

```powershell
.\.venv\Scripts\python.exe scripts/create_demo_dataset.py --output data/raw/demo
.\.venv\Scripts\python.exe scripts/audit_dataset.py --dataset data/raw/demo --output data/reports
```

Example final output for the intentional defects in this demo:

```text
Dataset Audit Complete
Images analyzed: 12 valid / 15 files
train: 4
val: 4
test: 4
Exact duplicate pairs: 3
Near duplicate pairs: 3
Cross-split leakage pairs: 5
Corrupted images: 2
Dataset Health: FAIL
Report: .../data/reports/<UTC timestamp>_<unique suffix>/report.md
```

Exit codes: **0** for PASS or WARNING, **1** for a completed audit classified FAIL,
and **2** for configuration, input, or filesystem errors. The demo intentionally exits 1.
Progress is logged to stderr and the final result is printed to stdout.

## Detection methods and limits

**Exact duplicates:** SHA-256 streams file bytes in 1 MiB chunks. Valid decoded images
with identical file hashes form a group; all unique unordered pairs are exported.
Different compression or metadata can produce different byte hashes for identical pixels.
Invalid files are excluded from duplicate detection.

**Near duplicates:** ImageHash computes a 64-bit pHash after EXIF orientation normalization
and RGB conversion. Hamming distance counts differing hash bits; pairs at or below the
configured threshold (default 5, inclusive, range 0–64) are candidates for manual review.
Byte-identical pairs are excluded from the near report. A BK-tree searches by Hamming
radius behind the `HammingIndex` interface. This prunes comparisons in favorable datasets,
but has no subquadratic worst-case guarantee; broad thresholds and matching-heavy datasets
can still generate quadratic time/output. Results are kept in memory in this first version.

Near groups are connected components of matching edges. A and C can share a group via B
without A and C matching directly. CSV rows contain only observed matching pairs, with
group ID and distance. Group IDs are deterministic for an unchanged dataset/configuration,
but should not be used as permanent identifiers across dataset versions.

pHash is a heuristic: low-detail images can collide, and crops/rotations can be missed.
It does not prove two images depict the same object. Only the first frame of animated or
multi-page images is analyzed. Metadata dimensions describe stored pixels; pHash alone
uses EXIF-normalized orientation. Pillow decoding limits remain enabled.

**Leakage:** each actual matching pair whose split names differ is exported with source,
target, split names, distance, and `exact_duplicate` or `near_duplicate` type.
This covers train–val, train–test, and val–test. Leakage can inflate evaluation quality
because validation/test examples may closely reproduce training data.

## Reports and statistics

Each run creates a unique UTC timestamped directory with:

| File | Contents |
| --- | --- |
| `summary.json` | Schema version, effective configuration, counts, health reasons, distributions |
| `images.csv` | Every inspected file, metadata, hashes, validation status and error |
| `exact_duplicates.csv` | All byte-identical image pairs and group IDs |
| `near_duplicates.csv` | All perceptual matching pairs, distances, and group IDs |
| `cross_split_leakage.csv` | Cross-split matching pairs and leakage type |
| `invalid_images.csv` | Unsupported, corrupt, unreadable, and zero-byte files |
| `report.md` | Readable overview, split sizes, counts, warnings, and failures |

All CSVs retain column headers when empty. Image path columns are relative to the dataset
root; decoder error messages can include absolute paths. Summary and Markdown reports
also record the absolute dataset location.
`total_files` counts every file inspected; `total_images` and split sizes count only valid
decoded images. Width, height, aspect ratio, and byte-size statistics use valid images:
count, mean, standard deviation, minimum, 25th percentile, median (`50%`), 75th percentile,
and maximum. Raw distributions are available in `images.csv`; empty statistics are `{}`.
For a single sample, standard deviation is undefined and serialized as `null`.

## Dataset health policy

Rules apply in precedence order:

1. **FAIL** if no valid images exist, any exact cross-split leakage exists, or the number of
   corrupt/unreadable/zero-byte files divided by **all inspected files** is strictly greater
   than `corrupted_image_failure_threshold` (default 0.01). Unsupported nonempty files are
   excluded from that numerator; they still generate warnings.
2. **WARNING** if any structural issue, empty split, invalid file, same-split exact duplicate,
   or perceptual matching pair exists. Also warn if any valid image has width or height below
   `minimum_dimension_warning` (default 32), or width/height above `aspect_ratio_warning`
   (default 10) or below its reciprocal. Equality does not trigger these dimension rules.
3. **PASS** otherwise. A PASS covers only implemented image-level checks.

General dimension variation is visible in statistics but is not automatically a warning.
Split ratios do not imply class imbalance; class imbalance checks await annotation support.
The report never deletes or rewrites images and never automatically resolves duplicates.

## Development and roadmap

Run `python -m pytest` and `python -m ruff check .` in the installed environment.
Tests generate images locally and require no external dataset or network.
Datasets, reports, virtual environments, secrets, model artifacts, and caches are ignored by Git.

- Phase 1 — Baseline object detection model
- Phase 2 — Training and experiment tracking
- Phase 3 — Web API
- Phase 4 — Web dashboard
- Phase 5 — Multimodal vision-language support
- Phase 6 — RAG over technical documentation
- Phase 7 — Automated inspection reports
- Phase 8 — MLOps/model monitoring
- Phase 9 — Production deployment

Suggested next GitHub Issues:

1. Add YOLO annotation validation: missing/orphan labels, class IDs, normalized boxes, tiny objects.
2. Add annotation-derived class balance summaries and configurable imbalance warnings.
3. Benchmark BK-tree matching and introduce streaming pair exports and disk-backed metadata.
4. Add a human review artifact for near matches with side-by-side thumbnails and decisions.
5. Add CI across supported Python versions/OSes and reproducible dependency constraints.
