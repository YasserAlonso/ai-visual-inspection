# UTS v3 annotation-quality adjudication

**Review date:** 2026-09-28. **Dataset:** `C:\Users\leoar\Desktop\utsv3_clean_final2` (read-only). **Training:** not run.

## Decision

**READY FOR BASELINE TRAINING WITH DOCUMENTED LABEL NOISE.** This is an authorization gate only; no model was trained in this review. The 266 Phase 0 tiny-box warnings were individually inspected in class-organized sheets. No annotation was clearly invalid. 206 were judged visually plausible; 60 remain questionable because the defect or its class cannot be confirmed at 640×640. No labels were changed.

The remaining uncertainty is concentrated in Dent and Rupture tiny boxes (all 57 are QUESTIONABLE), plus three Fastener Damage boxes. This is label uncertainty rather than evidence of severe systematic corruption. The 206 VALID tiny boxes are concentrated in Fastener Damage and usually mark visible dark features on seams, fasteners, or panel surfaces. The test split was viewed only for annotation integrity; no model metrics or development decisions were made from it.

## Tiny-box results

| Class | Reviewed | VALID | QUESTIONABLE | CLEARLY_INVALID |
|---|---:|---:|---:|---:|
| Dent | 53 | 0 | 53 | 0 |
| Fastener Damage | 209 | 206 | 3 | 0 |
| Rupture | 4 | 0 | 4 | 0 |
| **Total** | **266** | **206** | **60** | **0** |

QUESTIONABLE means the target/class cannot be confirmed from the processed image, including very small low-contrast marks or clipped targets. It does not assert that the source label is wrong. The repeated Dent examples on the same test image are visually indistinct at export resolution. No item was called CLEARLY_INVALID without strong evidence.

## Additional visual review

- **Broad/large boxes:** inspected 32 examples from the top-area sample of 353 boxes occupying at least 25% of the image. Some boxes encompass broad but visible damage regions; others are loose and include substantial surrounding panel. Large area alone was not treated as an error. This sample is purposive, not a census of all 353.
- **Overlap cases:** inspected both image-level cases at IoU ≥0.50 (same-class Rupture pairs; IoU 0.688 and 0.589). The overlap appears consistent with repeated/adjacent segmentation of elongated damage and is not enough to establish duplicate labels. No IoU ≥0.75 pairs were found; no boxes were removed.
- **Polygon-derived boxes:** inspected 32 examples from the 103 broad-envelope candidates among the 715 polygon-to-box conversions. Boxes generally cover the source damage, but axis-aligned envelopes can include considerable background for long or curved damage. This remains a known representation limitation; sample review does not adjudicate all 715 conversions.
- **Densely packed Fastener Damage:** the tiny-box sheets include repeated multi-annotation panel views. Visible target marks are generally aligned, but crowding, low contrast, and within-split repeats make some examples hard to distinguish.

These additional reviews were visual samples. Their individual annotations are identified by image, class, line where available, and group in the accompanying machine-readable records; they are not exhaustive label adjudication.

## Method and artifacts

The derivative was treated as read-only. Phase 0's `extremely_small_box` rows were joined to image/label lineage and rendered as manageable class-specific contact sheets. Each card shows the full image, box, enlarged crop, filename, split, class, and pixel dimensions. `annotation_review.csv` records all 266 tiny annotations with normalized/pixel coordinates, lineage and group reference, status, and notes. Candidate metrics and sheets cover large boxes, overlaps, and broad polygon-derived boxes. Inclusion on a candidate sheet is not itself a defect finding.

Artifacts are ignored by Git under `artifacts/audits/uts_v3_final_review/`:

- `annotation_review.csv`
- `tiny_boxes/Dent/`, `tiny_boxes/Fastener_Damage/`, `tiny_boxes/Rupture/`
- `large_boxes/`, `overlap_review/`, `broad_polygon_boxes/`
- `overlap_candidates.csv`, `polygon_envelope_candidates.csv`, `candidate_summary.json`

The tiny review manifest contains `VALID`, `QUESTIONABLE`, or `CLEARLY_INVALID`; `NOT_YET_REVIEWED` is reserved for pending records. No pending tiny records remain. The broader candidate sheets are sampled review aids, not a complete all-box manifest.

## Remaining concerns and Experiment #001

The inherited tiny labels, broad/overlapping boxes, polygon-envelope conversions, within-split repeats, limited source-photo provenance, and unresolved source class-wording inconsistency remain material caveats. The adjudication does not prove that every defect is labeled or that every box is tight.

**Recommendation:** proceed with AeroInspect Baseline Experiment #001 using only `utsv3_clean_final2`, after this review is recorded. Train from its existing train split and use validation only for the planned development process. Keep test sealed from training, threshold selection, hyperparameter decisions, and model selection; evaluate it only once after the model and evaluation procedure are frozen. Do not modify the source derivative as part of this adjudication. Record the documented label-noise caveat with experiment results.

No labels, images, splits, or model files were changed or created. No tests were run because no source code was changed. No commit, push, or release tag was made.
