# UTS Aircraft Defect Detection v3 clean derivative report

**Audit date:** 2026-09-28. **Training:** not run. **Source:** `C:\Users\leoar\Desktop\utsv3` (read-only). **Authoritative derivative:** `C:\Users\leoar\Desktop\utsv3_clean_final2`.

## Final decision

**APPROVE WITH WARNINGS** for a first object-detection baseline, after manual review of the flagged annotation examples below. The derivative contains only box-format labels, all source and near-duplicate groups remain within one split, every class is represented in every split, and the clean Phase 0 audit found no blocking annotation issues. The 266 inherited tiny boxes, repeated images within splits, broad/overlapping annotations, and unresolved source provenance/ontology wording remain material limitations.

No training, tuning, augmentation, oversampling, source editing, or test-set use for model selection was performed. The original archive and extracted source remain untouched. Earlier partial build attempts at `C:\Users\leoar\Desktop\utsv3_clean`, `...\utsv3_clean_reviewed`, and `...\utsv3_clean_final` are incomplete and must not be used; only `utsv3_clean_final2` is the audited derivative.

## Source and preparation

The source is Roboflow Universe Aircraft Defect Detection v3 (“No Nulls”), listed as Public Domain, with the class mapping `0 Dent`, `1 Fastener Damage`, `2 Rupture`. The original source audit rejected its original split because 25 exact duplicate pairs and 602 near-duplicate pairs crossed split boundaries, filename/capture groups spanned splits, and 715 polygon rows were mixed with 11,071 box rows. The global class balance was acceptable, but class composition differed across the original splits. The export README also conflicts with `data.yaml` on class wording, and original photo/annotation provenance is undocumented.

All **715** valid polygon rows were converted to one axis-aligned YOLO box using each polygon's normalized coordinate extrema (`xmin`, `xmax`, `ymin`, `ymax`). This converted 556 Dent and 159 Rupture polygons (no Fastener Damage polygons); source polygons had 5–234 points. Class IDs were preserved; all pre-existing five-value box rows were retained unchanged. Polygons with an extremum exactly at an image edge receive at most a `1e-9` normalized inward inset to prevent floating-point serialization from producing an out-of-bounds box. There were **0 invalid or unconvertible polygons**. The conversion report records class totals, point-count range, box-size distributions, and the numeric boundary treatment.

Source grouping joins SHA-256 exact matches, connected components of pHash matches at Hamming distance ≤5, canonicalized repeated filenames (Roboflow `.rf.<hash>` suffix removed), and exact `IMG_YYYYMMDD_HHMMSS` timestamp names. These are evidence-based source/capture indicators; no aircraft identity was inferred. The resulting **1,480 groups** (largest 16 images; median 6) were indivisible during splitting. Of these, 420 combine images from more than one old split.

A fixed seed (**42017**) drove a SciPy HiGHS mixed-integer assignment targeting 70/15/15 images and minimizing normalized absolute image-count and class-instance deviations, with each class constrained to appear in each split. The solver returned a feasible assignment at its time limit; the post-run constraint checks passed, but optimality was not proven. Actual split ratios are within 0.01 percentage point of target.

## Dataset counts and class balance

The derivative has 6,803 images and 6,803 corresponding label files, with no missing or orphan labels. All 11,786 annotation rows are now five-value YOLO boxes: 11,071 retained boxes plus 715 converted polygon boxes.

| Split | Images | % of images | Dent objects | Fastener Damage objects | Rupture objects | Total objects | Dent / Fastener / Rupture share of split objects |
|---|---:|---:|---:|---:|---:|---:|---|
| Train | 4,762 | 69.9985% | 2,907 | 2,686 | 2,658 | 8,251 | 35.23% / 32.55% / 32.21% |
| Validation (`valid`) | 1,020 | 14.9934% | 623 | 575 | 570 | 1,768 | 35.24% / 32.52% / 32.24% |
| Test | 1,021 | 15.0081% | 623 | 575 | 569 | 1,767 | 35.26% / 32.54% / 32.20% |
| **Total** | **6,803** | **100%** | **4,153** | **3,836** | **3,797** | **11,786** | **35.24% / 32.55% / 32.22%** |

The largest-to-smallest class ratio is **1.094:1** overall. The three class shares vary by less than 0.06 percentage point across splits; all are well represented in train, validation, and test. No duplication or oversampling was used to balance classes.

## Duplicate and leakage audit

| Grouping evidence | Source groups/pairs | Result in derivative |
|---|---:|---|
| SHA-256 exact | 481 exact duplicate pairs/groups | 481 pairs remain, all within one split; 0 cross-split pairs |
| pHash distance ≤5 | 4,030 near pairs across 2,223 connected components | 4,030 pairs remain, all within one split; 0 cross-split near pairs/groups |
| Canonical repeated filename | 1,053 groups; 396 crossed old splits | All consolidated before splitting |
| Exact timestamp filename | 861 groups; 296 crossed old splits | All consolidated before splitting |
| Combined evidence groups | **1,480** | No group crosses the new split boundaries |

Phase 0 independently rechecked exact and pHash matches after copying. It found **0 cross-split leakage pairs**. Exact and near-duplicate pairs remain within splits, so duplicated content can still overweight some examples during training. pHash matches are candidate similarities and can include false positives; the grouping is deliberately conservative.

## Annotation validation and visual review

The final Phase 0 run inspected all 6,803 images. It found no corrupt images, missing/orphan labels, malformed rows, invalid class IDs, non-finite values, out-of-range coordinates, zero-sized boxes, or out-of-bounds boxes. It returned **WARNING**, with **0 blocking annotation issues** and **266 `extremely_small_box` warnings**. The Phase 0 parser counts 11,520 accepted objects because it excludes those 266 tiny rows from its accepted-object statistics; the exact label-file count remains 11,786. The tiny rows comprise 2.26% of all instances and include small, low-contrast fastener annotations that need review.

Generated review sheets cover all three classes (up to 12 distinct image examples per class), polygon-to-box comparisons (up to 9 per polygon class), 18 distinct smallest-box images, 18 distinct largest-box images, and 18 distinct crowded/multiple-object images. Manual inspection found:

- Polygon-derived rectangles enclose their source polygons as expected, but thin or irregular shapes can produce broad boxes and lose shape detail.
- Several inherited Dent/Rupture boxes cover large skin panels or appear loose; densely packed Fastener Damage rows can overlap and become difficult to inspect.
- Some tiny/low-contrast targets are hard to distinguish at 640×640. Labels were not manually changed or removed.
- Exact duplicates and visually similar successive views are visible within splits. Grouping prevents split leakage, but these repeats remain a training redundancy risk.

This is a purposive visual sample, not exhaustive annotation adjudication. It cannot establish that all defects are labeled or that every box is semantically tight. The original class-description inconsistency and image-rights/source provenance remain unresolved.

## Files and reproducibility

- YOLO derivative: `C:\Users\leoar\Desktop\utsv3_clean_final2\` (`train/`, `valid/`, `test/`, `data.yaml`).
- Phase 0 report and CSVs: `C:\Users\leoar\Desktop\utsv3_clean_final2_audit\phase0\2026-09-28_210752_394333_9078ce68\`.
- Group manifest: `C:\Users\leoar\Desktop\utsv3_clean_final2_audit\group_manifest.csv`.
- Image/label lineage: `C:\Users\leoar\Desktop\utsv3_clean_final2_audit\dataset_lineage.csv`.
- Split assignment and balance: `C:\Users\leoar\Desktop\utsv3_clean_final2_audit\split_summary.json`.
- Polygon conversions: `C:\Users\leoar\Desktop\utsv3_clean_final2_audit\polygon_conversion_report.json`.
- Visual sheets: `C:\Users\leoar\Desktop\utsv3_clean_final2_audit\review_sheets\`.
- Source preservation and archive provenance remain documented in `docs/uts_v3_dataset_audit.md` and `data/external/uts_aircraft_defect_v3/dataset_provenance.json`.

## Next step

Manually adjudicate the 266 tiny-box warnings and the flagged broad/overlapping examples in the review sheets, then record that review against the lineage manifest. Keep the test split sealed. After that review is accepted, use this derivative for the first baseline; do not train until the review has been recorded.

**Adjudication completed:** see [UTS v3 annotation-quality adjudication](uts_v3_annotation_review.md) for the completed tiny-box review, sampled broad-box/overlap/polygon review, and training gate. The dataset files remain unchanged.
