# UTS Aircraft Defect Detection v3 dataset audit

**Audit date:** 2026-09-28. **Dataset path:** `C:\Users\leoar\Desktop\utsv3` (read-only). **Training:** not run.

## Final decision

**REJECT UNTIL FIXED** for baseline training.

Two independent blockers are present: (1) 627 duplicate/near-duplicate image pairs cross existing split boundaries, including 25 byte-identical train/validation/test pairs; and (2) 715 label rows are polygon segmentation records mixed into an otherwise box-formatted YOLO export. The current AeroInspect detection validator rejects those rows, and the split leakage makes validation/test performance unreliable. Global class balance is good, but class composition differs sharply by split. The source package also has unresolved label/provenance wording inconsistencies.

Do not train, rebalance, or edit this source export. The next step is to prepare a separate reviewed detection-only derivative or obtain a consistent box-only export, then make a group-aware split that keeps exact and confirmed near-duplicate/capture groups together. Preserve the original ZIP and extracted source unchanged. Re-audit the derivative before considering training.

## Dataset identity and provenance

- **Exact project/version:** Roboflow Universe, Aircraft Defect Detection v3, version name **No Nulls**, generated 2024-03-29. The export README says it was exported on 2025-01-27 10:57 GMT. [Version page](https://universe.roboflow.com/university-of-technology-sydney-21uto/aircraft-defect-detection/dataset/3)
- **Uploader:** University of Technology Sydney per Roboflow metadata. The included README says “Provided by a Roboflow user”; it does not identify the original image collectors or annotation authors.
- **Source URL:** `https://universe.roboflow.com/university-of-technology-sydney-21uto/aircraft-defect-detection/dataset/3`
- **License:** Roboflow and both included metadata files (`README.dataset.txt`, `data.yaml`) state **Public Domain**. The Roboflow listing links CC0 1.0. The ZIP contains no separate license file. The image rights/provenance beyond the uploader's stated license remain undocumented.
- **Archive:** `Aircraft Defect Detection.v3-no-nulls.yolov11.zip`; 275,867,489 bytes; SHA-256 `8DA00B298BD402BF829ADB9B7A311C8FADE1B8ECEF26FFEB64F0DDBF184E7237`. The recorded retrieval date, 2026-09-28, is inferred from filesystem modification time, not a separate download receipt. The archive was not modified.
- **Extracted working copy:** `C:\Users\leoar\Desktop\utsv3`, kept outside the repository and read-only. Archive/provenance are under ignored `data/external/uts_aircraft_defect_v3/`.
- **Verified `data.yaml` mapping:** `0 = Dent`, `1 = Fastener Damage`, `2 = Rupture`; `nc: 3`. Its `roboflow` metadata says workspace `university-of-technology-sydney-21uto`, project `aircraft-defect-detection`, version `3`, license `Public Domain`.
- **README inconsistency:** `README.roboflow.txt` describes labels as “dents-leaks-ruptures-other,” inconsistent with the 3-class `data.yaml`. The YOLO labels contain only IDs 0–2, so counts below follow `data.yaml`; the class ontology/provenance discrepancy remains unresolved.
- **Preprocessing:** auto-orient with EXIF orientation stripped; stretch resize to 640×640. README says no augmentation. All local image files are indeed 640×640. Native pre-export source resolution cannot be recovered from this export.

## Verified counts and annotation format

| Split | Images | Label files | Box-format rows | Polygon rows | Total annotated instances |
|---|---:|---:|---:|---:|---:|
| Train | 5,855 | 5,855 | 8,275 | 610 | 8,885 |
| Validation (`valid`) | 598 | 598 | 1,326 | 75 | 1,401 |
| Test | 350 | 350 | 1,470 | 30 | 1,500 |
| **Total** | **6,803** | **6,803** | **11,071** | **715** | **11,786** |

The 11,071 count is the exact number of five-value `class x_center y_center width height` rows across all splits. Another 715 rows have polygon-style YOLO segmentation shape: class ID followed by 6 or more normalized point coordinates. Those 715 rows are valid polygon records, not malformed numeric data, but they are incompatible with this project’s box-only annotation validator and must not be silently treated as bounding boxes.

- **Total image count:** 6,803, matching source metadata (train 5,855; valid 598; test 350).
- **Label files:** 6,803; each image has one corresponding label file. Missing labels: **0**. Orphan labels: **0**. Empty label files: **0**.
- **Bounding-box rows:** **11,071**. Of these, 266 are below the existing validator's tiny-box threshold and are warnings; 10,805 pass that configured area threshold.
- **Polygon annotation rows:** **715** (556 Dent, 159 Rupture, 0 Fastener Damage). They are geometrically valid normalized polygons, but the export mixes polygon and box labels.
- **Total annotation instances (boxes + polygons):** **11,786**. Images with zero annotated instances: **0**; exactly one instance: **4,563**; multiple instances: **2,240**; maximum: **41** instances in one image. Mean: **1.732 instances/image**. Mean box-format rows alone: **1.627/image**.
- Phase 0 reported 10,805 accepted objects because its box validator excludes both polygon rows and sub-threshold tiny boxes. That is not the total number of annotation records.

## Class distribution and balance

Percentages of all 11,071 box rows exclude polygon records. The “all annotated instances” column includes box and polygon records. Class image counts include either geometry; images may contain multiple classes, so image percentages sum above 100%.

| Class (ID) | Box rows | % of box rows | Polygon rows | All instances | % of all instances | Images containing class | % of all images | Train / valid / test instances* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dent (0) | 3,597 | 32.49% | 556 | 4,153 | 35.24% | 2,774 | 40.78% | 3,688 / 300 / 165 |
| Fastener Damage (1) | 3,836 | 34.65% | 0 | 3,836 | 32.55% | 1,711 | 25.15% | 2,196 / 656 / 984 |
| Rupture (2) | 3,638 | 32.86% | 159 | 3,797 | 32.22% | 2,916 | 42.86% | 3,001 / 445 / 351 |
| **Total** | **11,071** | **100%** | **715** | **11,786** | **100%** | — | — | **8,885 / 1,401 / 1,500** |

\*Per-split instances include both box and polygon rows.

Largest/smallest ratio is **1.094:1** over all annotated instances (4,153 / 3,797), or **1.066:1** using box rows only. Using a simple descriptive rule (≤1.5 reasonably balanced, >1.5–3 moderate, >3 severe), the *global* class balance is **reasonably balanced**; no global class is severely underrepresented.

That global result hides serious split drift. Among all annotated instances, Dent/Fastener/Rupture shares are approximately **41.5/24.7/33.8% train**, **21.4/46.8/31.8% validation**, and **11.0/65.6/23.4% test**. Fastener Damage dominates test while Dent is much less represented there. The split class mix is not stable enough for a clean comparative baseline.

## Annotation validation findings

Phase 0 was run on the external absolute path using an audit-specific config (`valid` folder name) and a derived YAML under ignored artifacts; the source `data.yaml` and dataset were not changed. The CLI returned **FAIL**.

| Check | Finding |
|---|---:|
| Image without label | 0 |
| Label without image | 0 |
| Empty label | 0 |
| Valid five-value box rows | 11,071 |
| Polygon rows where detector validator expects 5 values | 715 (all parse as valid polygons; format/task incompatibility) |
| Invalid class IDs | 0 |
| Non-numeric/non-finite values | 0 |
| Coordinates/width/height outside normalized range | 0 |
| Negative coordinates/sizes | 0 |
| Zero-width/height boxes | 0 |
| Boxes outside image bounds | 0 |
| Extremely small boxes (box area < 0.0001 normalized) | 266 |
| Other malformed numeric rows | 0 |

No labels were modified. The main structural issue is mixed annotation geometry, not corrupt coordinates. Tiny boxes are 2.40% of box rows; minimum box size reaches 2×1 pixels at 640×640, so those samples need manual review.

## Duplicate and leakage audit

Phase 0 used SHA-256 for exact matches and pHash Hamming distance threshold **5** for near matches.

| Match type | Within train | Within valid | Within test | Train↔valid | Train↔test | Valid↔test | Total pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exact SHA-256 | 454 | 0 | 2 | 20 | 1 | 4 | **481** |
| Near pHash (distance ≤5, exact hashes excluded) | 3,398 | 5 | 25 | 528 | 42 | 32 | **4,030** |

There are **25 exact cross-split pairs** (25 distinct exact groups), and **602 cross-split near pairs** across **396 distinct pHash groups**. Cross-split pHash distances range from 0 to 4; pHash threshold matches are candidates and not automatically all called leakage. Phase 0 reports 2,223 near groups overall (connected components; not all members of a group are directly similar).

All 25 exact cross-split pairs were included in exact-duplicate contact sheets and are byte-identical images. For near matches, 16 pairs from each cross-split partition (train/valid, train/test, valid/test; 48 sampled pairs total) were visually compared. The sampled pairs are overwhelmingly the same image or near-identical successive/re-encoded views, and many filenames match after removing Roboflow's `.rf.<hash>` suffix. This supports treating the cross-split groups as likely leakage until image-level review clears them. Representative sheets and full pair tables are in ignored `artifacts/audits/uts_v3/duplicate_review/` and the Phase 0 CSVs.

## Filename/source-group assessment

After stripping the Roboflow hash suffix and extension, **1,053 repeated canonical filename groups** were found; **396 groups span multiple splits**. Names include aircraft model/part identifiers such as `Cessna-170-Wing-Structure-Assembly...` and timestamped camera names such as `IMG_20230511_101356`. The latter occurs in both train and validation under different Roboflow hashes. Such names imply repeated source files or capture sequences; they do not prove aircraft identity or inspection session without source metadata. The exported metadata has no aircraft ID, capture-session ID, location, or sequence grouping. Existing splits therefore appear to separate some repeated captures/identical files.

## Image and box statistics

All 6,803 local images decode successfully and are **640×640** (min/median/mean/max width and height each 640; aspect ratio exactly 1.0). This confirms the stated processed size, not source-photo dimensions. File sizes: min **9,747 B**, Q1 **30,904 B**, median **39,830 B**, mean **39,956.9 B**, Q3 **48,181 B**, max **89,914 B**.

Box statistics use the 11,071 five-value detection rows only; polygons are excluded. Coordinates are normalized to the 640×640 image:

| Metric | Width | Height | Area (`w×h`) |
|---|---:|---:|---:|
| Minimum | 0.003125 | 0.0015625 | 0.0000146484 |
| Q1 | 0.0828125 | 0.078125 | 0.00651367 |
| Median | 0.159375 | 0.1484375 | 0.0241650 |
| Mean | 0.189404 | 0.169132 | 0.0481583 |
| Q3 | 0.242188 | 0.220313 | 0.0524316 |
| Maximum | 1.0 | 1.0 | 0.703049 |

For size bands, area thresholds are COCO-style side lengths applied to a 640×640 image: small `<(32/640)^2`, medium `(32/640)^2` to `<(96/640)^2`, large `≥(96/640)^2`.

| Size band | Count | % of box rows |
|---|---:|---:|
| Small | 1,760 | 15.89% |
| Medium | 3,535 | 31.92% |
| Large | 5,776 | 52.19% |

Maximum box area is 70.3% of the image; width and height each reach the full image extent. Such large boxes can be legitimate for broad damage but several sampled Dent/Rupture annotations look loose and need annotation-policy review.

## Visual review

Generated per-class overlay sheets for Dent, Fastener Damage, and Rupture. Each contains a seeded sample plus examples selected for smallest/largest annotation, multiple objects, low brightness, proximity to image edge, and a median-area typical example. Polygon rows are drawn as outlines; box rows as rectangles. Exact and near duplicate contact sheets are also present. Files are ignored audit artifacts:

- `artifacts/audits/uts_v3/review_sheets/Dent.jpg`
- `artifacts/audits/uts_v3/review_sheets/Fastener_Damage.jpg`
- `artifacts/audits/uts_v3/review_sheets/Rupture.jpg`
- `artifacts/audits/uts_v3/duplicate_review/`

Manual review of the 36 class-sheet examples found some tight fastener/crack annotations, but also broad boxes covering large skin panels, crowded overlapping labels, tiny/low-contrast targets, and class/geometry inconsistency. Dent/Rupture polygon examples coexist with rectangles. This is a purposive sample, not exhaustive adjudication; it cannot establish recall of all defects or box tightness over the full dataset. Test images were inspected only for dataset/label QA, not model development or tuning.

## Decision and next step

**REJECT UNTIL FIXED.** Exact cross-split image duplicates, high-confidence near-duplicate/capture groups across splits, 715 segmentation rows mixed with detection boxes, class-distribution drift, loose/tiny sampled boxes, and limited provenance prevent a trustworthy first detection baseline. Global class balance alone does not outweigh these issues.

Next, preserve this source export and create a separate candidate preparation copy. First resolve the 715 polygon records with an explicit, human-reviewed task/format decision (or obtain a consistent detection-box export); do not silently convert or drop them. Then consolidate identical and near-identical source/capture groups and create group-aware train/validation/test splits. Keep the test set sealed after the new split. Re-run the full audit and visual review on that derivative. Do not train until it passes.

## Audit artifacts and files

- Phase 0 report and exact/near/leakage CSVs: `artifacts/audits/uts_v3/phase0/2026-09-28_175459_157445_53821745/`
- Detailed content statistics: `artifacts/audits/uts_v3/content_statistics.json`
- Annotated class sheets and duplicate review sheets: `artifacts/audits/uts_v3/review_sheets/` and `artifacts/audits/uts_v3/duplicate_review/`
- Generated audit config/YAML and read-only analysis helpers: `artifacts/audits/uts_v3/`
- Provenance manifest: `data/external/uts_aircraft_defect_v3/dataset_provenance.json`

No tracked source code was changed, so no tests were run. No training, tuning, relabeling, rebalancing, split regeneration, commit, push, or release tag was performed.

## Sources

- UTS v3 metadata/version/export settings: https://universe.roboflow.com/university-of-technology-sydney-21uto/aircraft-defect-detection/dataset/3
- Included archive files: `README.dataset.txt`, `README.roboflow.txt`, `data.yaml`
- CC0 deed linked by Roboflow: https://creativecommons.org/publicdomain/zero/1.0/
