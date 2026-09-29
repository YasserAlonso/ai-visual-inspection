# Aircraft_Fuselage_DET2023 dataset record

## Verified status

The selected target is **Aircraft_Fuselage_DET2023: An Aircraft Fuselage Defect Detection Dataset**, listed in IEEE DataPort. The primary record URL is [IEEE DataPort](https://ieee-dataport.org/documents/aircraft-fuselagedet2023-aircraft-fuselage-defect-detection-dataset). The record is blocked by robots restrictions in this environment and there is no local package. Consequently, package-specific facts below remain unverified and no download, conversion, audit, split, analysis, or real training has been performed.

Bibliographic indexing associates the dataset with Xiaoyu Zhang, Jinping Zhang, Jiusheng Chen, Runxia Guo, and Jun Wu, IEEE DataPort, 2024. This metadata is discoverable through [DBLP's dataset record](https://dblp.org/rec/data/10/ZhangZCGW24), but the source organization, DOI, and direct terms for this particular record have not been confirmed. The related paper is [Zhang et al., IEEE Transactions on Artificial Intelligence, 5(7), 3551–3563 (2024), DOI 10.1109/TAI.2024.3372474](https://doi.org/10.1109/TAI.2024.3372474); this is a paper DOI, not a verified dataset DOI.

| Item | Current finding |
|---|---|
| License / usage terms | Unknown for this dataset record. Do not redistribute or use beyond terms shown on the official record/package. IEEE DataPort platform-level licensing summaries do not establish this dataset's specific license. |
| Image/object totals | Unverified. A third-party catalog reports 5,601 images and four categories; treat this only as a lead until reconciled with the package. |
| Annotation format | Unverified from an authoritative source. A community tutorial describes a Pascal VOC/XML folder; this is a lead, not a confirmed package specification. |
| Classes and definitions | Unverified. The user's intended semantics (corrosion, cracks, dents, missing/damaged fasteners) are not yet confirmed as the source labels. A secondary tutorial suggests source names `scratch`, `paint_peel`, `rust`, `rivet_damage`; do not assume these labels, order, or equivalence. |
| Class counts / resolutions | Unknown; measure from the acquired files. |
| Official train/validation/test split | Unknown. |
| Capture grouping / repeated surfaces | Unknown. The paper describes fuselage defect detection in complex environments, but available public metadata does not expose aircraft, region, inspection, or sequence IDs. |
| Limitations | Until source package/terms/metadata are inspected, dataset suitability, label coverage, domain representativeness, and leakage risk cannot be established. |

The dataset is conceptually an object-detection benchmark, but training suitability depends on the actual annotations containing usable boxes and a verified mapping. Do not claim a successful detector benchmark based solely on a catalog listing or conversion tutorial.

## Acquisition and preparation

Open the official IEEE DataPort record in a browser. Complete any registration/login and terms acceptance required by the record, then download the dataset and retain its supplied README/license/citation files. This agent could not inspect whether access is open, subscriber-only, or registration-gated. Place the untouched downloaded archive in `data/external/aircraft_fuselage_det2023/original/`. Do not commit it. Keep extracted working copies and all converted labels under `data/processed/aircraft_fuselage_det2023/`.

When the package is present, first inventory its archive/tree and source metadata, calculate SHA-256 checksums, and update this record with the exact source version, terms, counts, names, and annotation format. Then populate `configs/datasets/aircraft_fuselage_det2023.yaml` with the exact source classes and canonical names only after semantic review. Preserve the official split if one exists. Otherwise inspect identifiers and exact/perceptual duplicates before deterministic grouped splitting; record manifests under `data/manifests/aircraft_fuselage_det2023/`. Run the existing Phase 0 and annotation audits before using the prepared dataset with the baseline config.

The model is intended only to flag candidate regions for qualified human inspection review. It cannot establish airworthiness or replace certified aircraft inspectors.

## Preparation state

- Original dataset package: **not present**.
- Source archive checksums: **not generated**.
- Authoritative class mapping: **pending package inspection**.
- Processed annotation format, split counts, audit, duplicate/leakage findings, and analysis: **not available**.
- Real training and test-set evaluation: **not run**.
- Next step: acquire the package from the official record, observe its displayed terms, and place the untouched archive in the path above.

## Sources and confidence

- [Official IEEE DataPort dataset record](https://ieee-dataport.org/documents/aircraft-fuselagedet2023-aircraft-fuselage-defect-detection-dataset) — primary source; currently inaccessible to this environment.
- [DBLP bibliographic dataset entry](https://dblp.org/rec/data/10/ZhangZCGW24) — author/title/publication-year corroboration, not source terms or package metadata.
- [Related 2024 detection paper](https://doi.org/10.1109/TAI.2024.3372474) — task context, not a substitute for dataset package documentation.
- [Third-party catalog](https://www.selectdataset.com/dataset/8fbc5d0efc172a747e8493f99c95e69b) and [community Pascal VOC tutorial](https://ithelp.ithome.com.tw/articles/10338764) — discovery leads only; not used to assert canonical classes, licensing, or package structure.
