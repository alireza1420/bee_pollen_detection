"""
Sanity check — run this first, before anything else.

Tests the annotation converter on the sample JSON file you have,
draws the bounding boxes on the image, and saves a visualization
so you can visually confirm the conversion is correct.

Run with:
    python sanity_check.py
"""

import json
import cv2
import numpy as np
from pathlib import Path

# ── CONFIG: point these at your actual sample files ──────────────────────
JSON_PATH = '1780568190356_2022-03-26-10-00_002.json'
IMG_PATH  = '1780568217246_2022-03-26-10-00_002.jpg'
OUT_PATH  = 'sanity_check_result.jpg'
# ─────────────────────────────────────────────────────────────────────────

CLASS_MAP   = {'nonpollenbee': 0, 'pollenbee': 1}
CLASS_NAMES = {0: 'nonpollenbee', 1: 'pollenbee'}
COLORS      = {0: (0, 200, 255), 1: (0, 255, 80)}   # orange, green


def convert_one(json_path):
    """Return list of (class_id, cx, cy, w, h) — all normalized."""
    with open(json_path) as f:
        data = json.load(f)

    W, H = data['imageWidth'], data['imageHeight']
    results = []

    for shape in data['shapes']:
        label = shape['label']
        if label not in CLASS_MAP:
            print(f'  [WARN] Unknown label: {label}')
            continue

        cls_id = CLASS_MAP[label]
        pts    = shape['points']
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        x_min, x_max = max(0, min(xs)), min(W, max(xs))
        y_min, y_max = max(0, min(ys)), min(H, max(ys))

        cx = ((x_min + x_max) / 2) / W
        cy = ((y_min + y_max) / 2) / H
        w  = (x_max - x_min) / W
        h  = (y_max - y_min) / H

        results.append((cls_id, cx, cy, w, h))

    return results, W, H


def draw_boxes(img_path, annotations, orig_w, orig_h, out_path):
    """Draw the converted boxes on the image to verify visually."""
    img = cv2.imread(img_path)
    if img is None:
        print(f'[ERROR] Could not load image: {img_path}')
        return

    draw_h, draw_w = img.shape[:2]

    for cls_id, cx, cy, w, h in annotations:
        # De-normalize to pixel coordinates on the actual image
        px_cx = cx * draw_w
        px_cy = cy * draw_h
        px_w  = w  * draw_w
        px_h  = h  * draw_h

        x1 = int(px_cx - px_w / 2)
        y1 = int(px_cy - px_h / 2)
        x2 = int(px_cx + px_w / 2)
        y2 = int(px_cy + px_h / 2)

        color = COLORS[cls_id]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label_text = CLASS_NAMES[cls_id]
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label_text, (x1 + 2, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    cv2.imwrite(out_path, img)
    print(f'Visualization saved to: {out_path}')


# ── Run the check ──────────────────────────────────────────────────────────
print('=== Annotation Sanity Check ===\n')

annotations, orig_w, orig_h = convert_one(JSON_PATH)

print(f'Image size from JSON: {orig_w} x {orig_h}')
print(f'Total annotations:    {len(annotations)}')

counts = {}
for cls_id, *_ in annotations:
    counts[cls_id] = counts.get(cls_id, 0) + 1
for cls_id, cnt in counts.items():
    print(f'  Class {cls_id} ({CLASS_NAMES[cls_id]}): {cnt} instances')

print('\nFirst 3 converted annotations (YOLO format):')
print(f'  {"cls":>4}  {"cx":>8}  {"cy":>8}  {"w":>8}  {"h":>8}')
for cls_id, cx, cy, w, h in annotations[:3]:
    print(f'  {cls_id:>4}  {cx:>8.4f}  {cy:>8.4f}  {w:>8.4f}  {h:>8.4f}')

print('\nChecking all values are in [0, 1]:')
all_valid = True
for i, (cls_id, cx, cy, w, h) in enumerate(annotations):
    if not all(0 <= v <= 1 for v in [cx, cy, w, h]):
        print(f'  [FAIL] Annotation {i} has out-of-range values: cx={cx} cy={cy} w={w} h={h}')
        all_valid = False
if all_valid:
    print('  All values in [0, 1] ✓')

print('\nDrawing boxes on image for visual verification...')
draw_boxes(IMG_PATH, annotations, orig_w, orig_h, OUT_PATH)
print('\nDone. Open sanity_check_result.jpg to visually confirm the boxes look correct.')
