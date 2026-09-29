# AeroInspect AI: first aerospace detection dataset comparison

**Research snapshot:** 2026-09-28. Metadata/source review only. No archives downloaded and no model trained. Listings can change; recheck selected version and terms at acquisition.

## Recommendation

**Recommend University of Technology Sydney (UTS) Aircraft Defect Detection, specifically version 3 (6,803 images), as the first YOLO baseline candidate.** It is explicitly object detection, has an identifiable university uploader, lists a Public Domain license, supports YOLO export, and v3 reports no generated augmentation. Use the versioned [v3 export](https://universe.roboflow.com/university-of-technology-sydney-21uto/aircraft-defect-detection/dataset/3), not the 9,352-image project gallery or augmented v2.

This is conditional: public metadata does not give per-class object counts, total boxes, class image counts, or original photo provenance. Before training, acquire only v3 if its terms remain clear, run the repository's dataset/duplicate audit, calculate class histograms, and manually review a stratified sample from every class. If labels or split integrity fail, do not train.

## Comparison summary

| Dataset | Images | Object instances | Classes | Smallest class count | Largest class count | Balance quality | License | Provenance quality | YOLO compatibility | Duplicate risk | Main strength | Main weakness |
|---|---:|---:|---|---:|---:|---|---|---|---|---|---|---|
| [Aircraft Defect Detection (UTS), v3](https://universe.roboflow.com/university-of-technology-sydney-21uto/aircraft-defect-detection/dataset/3) | 6,803 v3; gallery 9,352 | Unknown | Dent; Fastener Damage; Rupture | Unknown | Unknown | Unknown | Public Domain per Roboflow | Medium: institutional uploader; original capture and labeling undocumented | Medium/unknown: no generated augmentations in v3; repeats/overlap unverified | Aerospace detection labels, identifiable uploader, permissive terms stated | No class/object counts or verified label quality |
| [aircraftsurface1, v1](https://universe.roboflow.com/yolo11aircraft/aircraftsurface1/dataset/1) | 5,041 gallery; 11,091 export | Unknown | Crack; Scratch; Dent; Corrosion; Missing-Head; Paint-off; Repair | Unknown | Unknown | Unknown | CC BY 4.0 per Roboflow | Yes; YOLO exports offered | High: 3 outputs/training example plus crop/rotation; source overlap unknown | Surface defect taxonomy, YOLO-ready | Count inflated by generated training images; provenance and class counts absent |
| [Aircraft_Defect, hediatma](https://universe.roboflow.com/hediatma/aircraft-defect) / [astika](https://universe.roboflow.com/astika/aircraft-defect) | About 2,650 each per search listing | Not available; surfaced projects are classification | Corrosion; Crack; Dent; Missing Head; Paint Off; Scratch | Unknown | Unknown | Unknown | Unknown | Low: individual uploaders; source/label process undocumented | No verified YOLO detection labels; cross-project duplication unknown | Six aircraft-skin categories | Not a verified object-detection dataset; unsuitable as-is for YOLO |

**Candidate ambiguity:** “Aircraft_Defect (~2,650)” resolves in Roboflow search to two same-named six-class **classification** projects (hediatma and astika). Their relation is unknown. Neither is treated as a box-annotated detection dataset. If another project was intended, its exact URL is needed.

## Dataset records

### 1. Aircraft Defect Detection — University of Technology Sydney

- **Exact name/source:** Aircraft Defect Detection; [project](https://universe.roboflow.com/university-of-technology-sydney-21uto/aircraft-defect); selected [v3 page](https://universe.roboflow.com/university-of-technology-sydney-21uto/aircraft-defect-detection/dataset/3).
- **Owner/uploader:** University of Technology Sydney, per Roboflow.
- **Original source/provenance:** Unknown. Project has no description; image collection and annotation process undocumented.
- **License/commercial implications:** Roboflow says Public Domain and links [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). Clearest commercial-use signal among candidates, but rights/authorization for every underlying image are not explained. Verify terms on the actual export.
- **Images:** Project gallery 9,352; two versions. v3: **6,803** (train 5,855 / validation 598 / test 350). v2: **25,736** (train 23,604 / validation 1,183 / test 949). These differ because v2 creates 3 outputs per training example and applies flips, crops, rotations, brightness changes, and mosaic. v1 details were not inspected. Use v3 counts.
- **Download size:** Unknown; no archive downloaded and public metadata gave no size.
- **Classes:** `Dent`, `Fastener Damage`, `Rupture`.
- **Images per class / object instances and percentages:** Unknown; pages show no class counts or total box count. Balance and underrepresented classes cannot be assessed.
- **Augmentation/derived imagery:** v3 explicitly says no augmentations. v2 explicitly applies augmentation. Whether gallery images themselves include prior-derived copies is unknown.
- **Splits:** v3 train 5,855 / validation 598 / test 350. Split methodology and grouping by aircraft/capture session unknown.
- **Annotation format / YOLO:** Object detection; page offers YOLOv5–YOLO26 exports (TXT labels and YAML). Integrity not checked.
- **Resolution:** v3 auto-orients and stretches to 640×640; native source resolutions unknown.
- **Multiple defects/image:** Unknown.
- **Duplicates/near-duplicates:** Unknown. v3 has no generated augmentation, but no hash or visual audit was run; repeat captures remain possible.
- **Quality/limitations:** Box tightness, omissions, tiny defects, and mislabeled images cannot be assessed without files. “Fastener Damage” is undefined. No class histogram, collection protocol, or original source attribution is published.

### 2. aircraftsurface1 — Yolo11aircraft

- **Exact name/source:** aircraftsurface1; [project](https://universe.roboflow.com/yolo11aircraft/aircraftsurface1); [v1 export](https://universe.roboflow.com/yolo11aircraft/aircraftsurface1/dataset/1).
- **Owner/uploader:** `Yolo11aircraft`, Roboflow workspace/author.
- **Original provenance:** Unknown; no description or collection/annotation method surfaced.
- **License/commercial implications:** Roboflow states [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), permitting commercial reuse subject to attribution and conditions. Uploader's rights to all source photos not established.
- **Images:** Gallery 5,041; v1 export **11,091** (train 9,075 / validation 1,008 / test 1,008). Export says 3 outputs per training example with crop up to 20% and rotation ±15%, so it includes generated training images. Original unique image count is not given.
- **Download size:** Unknown; not downloaded and no public size found.
- **Classes:** `Crack`, `Scratch`, `Dent`, `Corrosion`, `Missing-Head`, `Paint-off`, `Repair`.
- **Per-class image and object counts/percentages:** Unknown; no histogram or total box count shown. Underrepresentation cannot be assessed.
- **Splits:** v1 train 9,075 / validation 1,008 / test 1,008 (82/9/9 rounded). Whether derivatives were split with source images or after split is undocumented, creating leakage risk.
- **Augmentation/duplicates:** Explicit crops and rotations, 3 outputs/training example. Number of source originals unknown. Cross-split duplicate audit not performed.
- **Format/YOLO:** Object detection; YOLOv5, v8, v9, v11, v12, v26 among offered exports.
- **Resolution:** Auto-oriented, stretched to 640×640; source dimensions unknown.
- **Multiple defects/image:** Unknown.
- **Quality/limitations:** Cannot inspect box consistency or missing labels before file access. `Repair` may describe a repair state/action rather than a physical defect; semantics may overlap. Model metrics were not used as quality evidence.

### 3. Aircraft_Defect — approximately 2,650 images

- **Identity/source:** Two projects surfaced: [hediatma](https://universe.roboflow.com/hediatma/aircraft-defect) and [astika](https://universe.roboflow.com/astika/aircraft-defect). Search lists each at about 2,650 images and identifies both as classification with six categories. Exact current version counts and relationship are unknown.
- **Owner/uploader:** hediatma and astika respectively.
- **Original source/provenance:** Unknown; no collection/annotation process found.
- **License/commercial use:** Unknown from authoritative project details reviewed. Do not presume commercial rights.
- **Images/splits/size/resolution:** About 2,650 from search snippets; exact version totals, split, archive size, dimensions, unique-image count unknown.
- **Classes:** `Corrosion`, `Crack`, `Dent`, `Missing Head`, `Paint Off`, `Scratch` (names normalized from listing).
- **Class images/object instances/percentages:** Unknown; no object boxes or histogram established.
- **Task/format/YOLO:** Surfaced projects are image classification, not verified object detection. No box format or YOLO detection export established.
- **Augmentation/duplicates/multiple defects:** Unknown. Image labels do not establish object counts or defects per image.
- **Quality/limitations:** No boxes to assess. Box looseness, missing boxes, tiny objects, and multi-instance behavior cannot be evaluated. Class definitions and original provenance undocumented.

## Class balance

| Dataset/class | Object instances | % of all objects | Assessment |
|---|---:|---:|---|
| UTS v3 — Dent | Unknown | Unknown | Representation cannot be assessed |
| UTS v3 — Fastener Damage | Unknown | Unknown | Representation cannot be assessed |
| UTS v3 — Rupture | Unknown | Unknown | Representation cannot be assessed |
| aircraftsurface1 — Crack | Unknown | Unknown | Representation cannot be assessed |
| aircraftsurface1 — Scratch | Unknown | Unknown | Representation cannot be assessed |
| aircraftsurface1 — Dent | Unknown | Unknown | Representation cannot be assessed |
| aircraftsurface1 — Corrosion | Unknown | Unknown | Representation cannot be assessed |
| aircraftsurface1 — Missing-Head | Unknown | Unknown | Representation cannot be assessed |
| aircraftsurface1 — Paint-off | Unknown | Unknown | Representation cannot be assessed |
| aircraftsurface1 — Repair | Unknown | Unknown | Representation cannot be assessed |
| Aircraft_Defect — six classification categories | Unknown | Unknown | No detection object counts; class image counts unavailable |

**Severely underrepresented classes:** Not determinable from public metadata. Image totals must not be substituted for object counts. Extract image and instance histograms before accepting any dataset.

## Why UTS v3 over the alternatives

UTS v3 combines aircraft-defect detection, direct YOLO export, a documented versioned split, an identifiable university uploader, and a Public Domain listing. Unlike UTS v2 and aircraftsurface1 v1, it reports no generated augmentation. Fewer classes are acceptable for a coherent first baseline.

This does **not** establish superior labels: no object-count breakdown or source-image inspection is public. aircraftsurface1 has broader surface labels and CC BY 4.0, but its listed export is augmentation-inflated, split grouping undocumented, and provenance/class semantics thin. The ~2,650 Aircraft_Defect listings are weaker because they are classification projects, not verified box labels, and have unclear terms/origins.

## Exact next step before training

Acquire **only UTS Aircraft Defect Detection v3** after confirming Public Domain terms on the actual download page. Keep the untouched archive in ignored `data/external/`; record source URL, version, retrieval date, license text, byte size, and SHA-256. Extract to a temporary audit location and run the existing audit before training. Produce exact per-class image and instance histograms, split counts, original-resolution distribution, and visual box-review sheets for every class. Check exact/perceptual duplicates across splits and whether augmented siblings cross splits. Stop if terms, provenance, labels, or leakage fail review; only then consider the existing YOLO baseline configuration.

## Sources

- UTS project: https://universe.roboflow.com/university-of-technology-sydney-21uto/aircraft-defect-detection
- UTS v2 splits, preprocessing and augmentations: https://universe.roboflow.com/university-of-technology-sydney-21uto/aircraft-defect-detection/dataset/2
- UTS v3 splits, preprocessing and no-augmentation statement: https://universe.roboflow.com/university-of-technology-sydney-21uto/aircraft-defect-detection/dataset/3
- aircraftsurface1 project: https://universe.roboflow.com/yolo11aircraft/aircraftsurface1
- aircraftsurface1 v1 export: https://universe.roboflow.com/yolo11aircraft/aircraftsurface1/dataset/1
- Aircraft_Defect (hediatma): https://universe.roboflow.com/hediatma/aircraft-defect
- Aircraft_Defect (astika): https://universe.roboflow.com/astika/aircraft-defect
