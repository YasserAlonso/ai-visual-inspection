"""Create validation-only qualitative detection error sheets."""

from pathlib import Path

from PIL import Image, ImageDraw


def generate_validation_error_examples(model, dataset, output: Path, *, device: str = "0") -> dict:
    """Save small validation-only examples of matches and common detection errors."""
    image_dir = dataset.splits["val"]
    categories = {
        "true_positives": [], "false_positives": [], "false_negatives": [],
        "low_confidence": [], "class_confusion": [], "small_object_misses": [],
        "crowded_fastener": [], "broad_dent_rupture": [],
    }
    image_paths = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    predictions = model.predict(
        source=[str(p) for p in image_paths], imgsz=640, conf=0.001, device=device,
        stream=True, verbose=False,
    )
    names = dataset.names

    def iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        return inter / max(area_a + area_b - inter, 1e-12)

    for path, result in zip(image_paths, predictions):
        label_path = path.parent.parent / "labels" / f"{path.stem}.txt"
        gt = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                c, xc, yc, w, h = map(float, line.split())
                gt.append((int(c), ((xc - w / 2) * 640, (yc - h / 2) * 640,
                                    (xc + w / 2) * 640, (yc + h / 2) * 640)))
        pred = []
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy().astype(int)
            pred = [(int(c), tuple(map(float, box)), float(score)) for c, box, score in zip(cls, xyxy, conf)]
        matched_gt, matched_pred = set(), set()
        for pi in sorted(range(len(pred)), key=lambda i: pred[i][2], reverse=True):
            c, box, score = pred[pi]
            options = [(iou(box, gbox), gi) for gi, (gc, gbox) in enumerate(gt) if gc == c and gi not in matched_gt]
            if options and max(options)[0] >= 0.5:
                _, gi = max(options)
                matched_gt.add(gi)
                matched_pred.add(pi)
                categories["true_positives"].append((path, gt, pred, f"TP {names[c]} {score:.2f}"))
            else:
                categories["false_positives"].append((path, gt, pred, f"FP {names[c]} {score:.2f}"))
                wrong = [(iou(box, gbox), gc) for gc, gbox in gt if gc != c]
                if wrong and max(wrong)[0] >= 0.5:
                    categories["class_confusion"].append((path, gt, pred, f"Class confusion: predicted {names[c]}"))
            if score < 0.25:
                categories["low_confidence"].append((path, gt, pred, f"Low confidence {names[c]} {score:.2f}"))
        for gi, (c, box) in enumerate(gt):
            if gi not in matched_gt:
                categories["false_negatives"].append((path, gt, pred, f"FN {names[c]}"))
                if (box[2] - box[0]) * (box[3] - box[1]) < 32 * 32:
                    categories["small_object_misses"].append((path, gt, pred, f"Small miss {names[c]}"))
        if sum(c == 1 for c, _ in gt) >= 5:
            categories["crowded_fastener"].append((path, gt, pred, "Crowded Fastener Damage"))
        for c, box in gt:
            if c in (0, 2) and (box[2] - box[0]) * (box[3] - box[1]) >= 0.25 * 640 * 640:
                categories["broad_dent_rupture"].append((path, gt, pred, f"Broad {names[c]}"))
                break

    counts = {}
    output.mkdir(parents=True, exist_ok=True)
    for category, examples in categories.items():
        selected, seen = [], set()
        for example in examples:
            if example[0] not in seen:
                selected.append(example)
                seen.add(example[0])
            if len(selected) >= 12:
                break
        counts[category] = {"available": len(examples), "saved": len(selected)}
        if not selected:
            continue
        cards = []
        for path, gt, pred, title in selected:
            im = Image.open(path).convert("RGB").resize((640, 640))
            draw = ImageDraw.Draw(im)
            for c, box in gt:
                draw.rectangle(box, outline=(40, 210, 40), width=3)
            for c, box, score in pred:
                draw.rectangle(box, outline=(255, 45, 45), width=2)
            card = Image.new("RGB", (660, 690), "white")
            card.paste(im, (10, 10))
            ImageDraw.Draw(card).text((10, 655), f"{title} | {path.name}", fill="black")
            cards.append(card)
        sheet = Image.new("RGB", (1320, ((len(cards) + 1) // 2) * 690), "#ddd")
        for i, card in enumerate(cards):
            sheet.paste(card, ((i % 2) * 660, (i // 2) * 690))
        sheet.save(output / f"{category}.jpg", quality=92)
    return counts
