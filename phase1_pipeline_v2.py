"""
Phase 1 — YOLOv8s Baseline Pipeline for VnPollenBee
=====================================================

Dataset folder structure (as downloaded):
    pollenbees/
      train/
        annotations_only_non_and_pollenbee_fixed_final/  <- Labelme JSON files
        images/                                           <- JPG files
      val/
        annotations_only_non_and_pollenbee_fixed_final/
        images/
      test/
        annotations_only_non_and_pollenbee_fixed_final/
        images/

What this file does, in order:
  1. Converts all Labelme JSON files to YOLO .txt format (run once, saved in-place)
  2. Defines PollenBeeDataset — loads images + converted labels
  3. Defines collate_fn — handles variable number of boxes per image
  4. Defines train_one_epoch — one full pass over training data
  5. Defines evaluate — computes detection mAP using torchmetrics
  6. main() — ties everything together with logging and checkpointing
"""

import os, json, glob, random, logging, csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Logging setup — timestamps + level so you can trace exactly what happened
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)


# ===========================================================================
# PART 1 — Constants and folder helpers
# ===========================================================================

# Class mapping — explicit so there is zero ambiguity in the paper.
# nonpollenbee = 0  (majority class, ~59 000 instances)
# pollenbee    = 1  (minority class, ~1 758 instances)
CLASS_MAP = {
    'nonpollenbee': 0,
    'pollenbee':    1,
}
CLASS_NAMES = {v: k for k, v in CLASS_MAP.items()}   # reverse lookup

# The long annotations folder name — defined once here so if it ever
# changes you only need to update this single line.
ANNOTATIONS_FOLDER = 'annotations_only_non_and_pollenbee_fixed_final'


def annotations_dir(split_root: str) -> str:
    """Return the full path to the annotations folder for a given split root."""
    return os.path.join(split_root, ANNOTATIONS_FOLDER)


def images_dir(split_root: str) -> str:
    """Return the full path to the images folder for a given split root."""
    return os.path.join(split_root, 'images')


def labels_dir(split_root: str) -> str:
    """
    Return the path where converted YOLO .txt files will be written.
    We create a 'labels/' folder alongside 'images/' and 'annotations_only_.../'
    so the structure stays clean and the conversion can be skipped on future runs.
    """
    return os.path.join(split_root, 'labels')


# ===========================================================================
# PART 2 — Annotation Converter  (Labelme JSON → YOLO .txt)
# ===========================================================================
# This conversion only needs to run once.  After the first run, the labels/
# folder exists and we skip conversion entirely.

def convert_one_json(json_path: str, out_dir: str) -> str:
    """
    Convert a single Labelme JSON file to a YOLO .txt file.

    Conversion formula for each bounding box:
        Given raw pixel points [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]:

        x_min = min(x0..x3),  x_max = max(x0..x3)
        y_min = min(y0..y3),  y_max = max(y0..y3)

        cx = (x_min + x_max) / 2  /  image_width     # normalised centre-x
        cy = (y_min + y_max) / 2  /  image_height    # normalised centre-y
        w  = (x_max - x_min)      /  image_width     # normalised width
        h  = (y_max - y_min)      /  image_height    # normalised height

    Why use min/max instead of assuming a fixed corner order?
    Labelme polygons can start from any corner.  Taking the axis-aligned
    bounding box handles all orderings safely.
    """
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    img_w = data['imageWidth']
    img_h = data['imageHeight']

    os.makedirs(out_dir, exist_ok=True)
    stem     = Path(json_path).stem
    out_path = os.path.join(out_dir, stem + '.txt')

    lines = []
    for shape in data['shapes']:
        label = shape['label']

        if label not in CLASS_MAP:
            log.warning(f'Unknown label "{label}" in {json_path} — skipping shape.')
            continue

        cls_id = CLASS_MAP[label]
        pts    = shape['points']          # list of [x, y]

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        # Clamp to image bounds — annotations occasionally extend 1-2px outside
        x_min = max(0.0, min(xs))
        y_min = max(0.0, min(ys))
        x_max = min(float(img_w), max(xs))
        y_max = min(float(img_h), max(ys))

        # Skip degenerate (zero-area) boxes
        if (x_max - x_min) <= 0 or (y_max - y_min) <= 0:
            log.warning(f'Degenerate box in {json_path} — skipping.')
            continue

        cx = ((x_min + x_max) / 2.0) / img_w
        cy = ((y_min + y_max) / 2.0) / img_h
        w  = (x_max - x_min)         / img_w
        h  = (y_max - y_min)         / img_h

        lines.append(f'{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))

    return out_path


def convert_split(split_root: str):
    """
    Convert all JSON annotations for one split (train / val / test).

    If the labels/ folder already exists and has the same number of .txt
    files as there are .json files, conversion is skipped — this means
    you can re-run this script many times without re-doing the work.
    """
    ann_dir = annotations_dir(split_root)
    lbl_dir = labels_dir(split_root)

    if not os.path.isdir(ann_dir):
        log.warning(f'Annotations folder not found: {ann_dir}  — skipping.')
        return

    # The dataset also ships split-level summary files such as
    # pollenbee_train.json. Those are not per-image Labelme annotations.
    json_files = sorted([
        p for p in glob.glob(os.path.join(ann_dir, '*.json'))
        if not Path(p).name.startswith('pollenbee_')
    ])

    # Skip if already converted (same file count = safe to skip)
    if os.path.isdir(lbl_dir):
        existing = glob.glob(os.path.join(lbl_dir, '*.txt'))
        if len(existing) == len(json_files):
            log.info(f'Labels already exist for {split_root} ({len(existing)} files) — skipping conversion.')
            return

    log.info(f'Converting {len(json_files)} JSON files in {ann_dir} ...')
    for jf in json_files:
        convert_one_json(jf, lbl_dir)
    log.info(f'Conversion done → {lbl_dir}')


# ===========================================================================
# PART 3 — Dataset Class
# ===========================================================================

class PollenBeeDataset(Dataset):
    """
    Loads images and YOLO-format labels for one split of VnPollenBee.

    After convert_split() has run, each split root will contain:
        split_root/
          images/   *.jpg
          labels/   *.txt   (same stems as the images)

    A .txt file has one line per bounding box:
        class_id  cx  cy  w  h   (all normalised to [0, 1])

    Images with no annotated bees have an empty .txt file — we return
    empty tensors for those, which the collate_fn handles gracefully.
    """

    def __init__(self, split_root: str, img_size: int = 640):
        self.img_size = img_size
        self.img_dir  = images_dir(split_root)
        self.lbl_dir  = labels_dir(split_root)

        self.img_paths = sorted([
            p for p in glob.glob(os.path.join(self.img_dir, '*'))
            if p.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        if not self.img_paths:
            raise FileNotFoundError(f'No images found in {self.img_dir}')

        log.info(f'Loaded {len(self.img_paths)} images from {self.img_dir}')
        self._log_class_distribution()

    def _label_path(self, img_path: str) -> str:
        """Derive the .txt label path from an image path."""
        return os.path.join(self.lbl_dir, Path(img_path).stem + '.txt')

    def _log_class_distribution(self):
        """
        Count instances per class across the split.
        This is the number you will report in the paper to characterise
        the imbalance (expected ~1:33 ratio of pollenbee:nonpollenbee).
        """
        counts = {0: 0, 1: 0}
        for img_path in self.img_paths:
            lbl = self._label_path(img_path)
            if os.path.exists(lbl):
                with open(lbl) as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            cls = int(parts[0])
                            counts[cls] = counts.get(cls, 0) + 1
        total = sum(counts.values())
        for cls_id, cnt in counts.items():
            log.info(
                f'  {CLASS_NAMES.get(cls_id, cls_id)}: {cnt} instances '
                f'({100 * cnt / max(total, 1):.1f}%)'
            )
        # Store for the loss (effective-number class weighting reads these).
        self.class_counts = counts

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]

        # ── Load and preprocess image ────────────────────────────────────
        img = cv2.imread(img_path)
        if img is None:
            raise IOError(f'Could not read: {img_path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Simple resize to square — for the baseline (Phase 1) we deliberately
        # use the simplest possible preprocessing so the baseline is clean.
        # Phase 2 can add letterboxing without changing the architecture.
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)   # HWC → CHW

        # ── Load labels ──────────────────────────────────────────────────
        boxes, classes = [], []
        lbl_path = self._label_path(img_path)

        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id      = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:])
                        boxes.append([cx, cy, w, h])
                        classes.append(cls_id)

        if boxes:
            boxes_t   = torch.tensor(boxes,   dtype=torch.float32)            # (N, 4)
            classes_t = torch.tensor(classes, dtype=torch.float32).unsqueeze(1)  # (N, 1)
        else:
            boxes_t   = torch.zeros((0, 4), dtype=torch.float32)
            classes_t = torch.zeros((0, 1), dtype=torch.float32)

        return img, boxes_t, classes_t


# ===========================================================================
# PART 4 — Collate Function
# ===========================================================================
# PyTorch's default collate_fn stacks tensors into a batch.
# That works for images (all the same size after resize), but fails for
# labels because different images have different numbers of boxes.
#
# Solution: flatten all boxes from all images into one big tensor, and
# carry an 'idx' tensor that records which image each box belongs to.
# This is exactly the format your existing ComputeLoss already expects.
#
# Example — batch of 3 images with 14, 0, and 3 boxes:
#   box shape : (17, 4)
#   cls shape : (17, 1)
#   idx values: [0,0,...,0,  2,2,2]   (image 1 has 0 boxes → no entries)

def collate_fn(batch):
    imgs, all_boxes, all_classes = zip(*batch)
    imgs = torch.stack(imgs, dim=0)   # (B, C, H, W) — all same size ✓

    box_list, cls_list, idx_list = [], [], []
    for i, (boxes, classes) in enumerate(zip(all_boxes, all_classes)):
        n = boxes.shape[0]
        if n > 0:
            box_list.append(boxes)
            cls_list.append(classes)
            idx_list.append(torch.full((n,), i, dtype=torch.long))

    if box_list:
        targets = {
            'box': torch.cat(box_list, dim=0),   # (total_N, 4)
            'cls': torch.cat(cls_list, dim=0),   # (total_N, 1)
            'idx': torch.cat(idx_list, dim=0),   # (total_N,)
        }
    else:
        targets = {
            'box': torch.zeros((0, 4)),
            'cls': torch.zeros((0, 1)),
            'idx': torch.zeros((0,), dtype=torch.long),
        }

    return imgs, targets


# ===========================================================================
# PART 5 — Training Loop
# ===========================================================================

def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, epoch):
    """
    One full pass over the training data.
    Returns the average combined loss (cls + box + dfl) across all batches.
    """
    model.train()
    total_loss = 0.0

    for batch_idx, (imgs, targets) in enumerate(loader):
        imgs               = imgs.to(device)
        targets['box']     = targets['box'].to(device)
        targets['cls']     = targets['cls'].to(device)
        targets['idx']     = targets['idx'].to(device)

        outputs            = model(imgs)
        loss_parts         = criterion(outputs, targets)   # (cls_loss, box_loss, dfl_loss)
        loss               = sum(loss_parts)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping prevents exploding gradients — especially
        # important in the first few epochs on a small dataset.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()
        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            cls_l, box_l, dfl_l = [x.item() for x in loss_parts]
            log.info(
                f'  Epoch {epoch} [{batch_idx+1}/{len(loader)}] '
                f'total={loss.item():.4f}  cls={cls_l:.4f}  box={box_l:.4f}  dfl={dfl_l:.4f}'
            )

    scheduler.step()
    return total_loss / len(loader)


# ===========================================================================
# PART 6 — Evaluation Loop
# ===========================================================================

def cxcywh_to_xyxy(boxes):
    """
    Convert bounding box format from [cx, cy, w, h] to [x1, y1, x2, y2].
    This is needed because torchmetrics and torchvision NMS both expect xyxy.
    """
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=1)


def non_max_suppression(pred_batch, conf_threshold=0.001, iou_threshold=0.70, max_det=300, max_nms=30000):
    """
    Filter and de-duplicate detections from the model's inference output.

    At inference time, your Head returns a tensor of shape:
        (batch_size, 4 + num_classes, num_anchors)
    where columns 0:4 are decoded [cx, cy, w, h] in pixel space, and
    columns 4: are sigmoid class probabilities.

    NMS keeps only the highest-confidence detection in each overlapping group.
    """
    from torchvision.ops import batched_nms

    results = []
    for pred in pred_batch:
        pred = pred.T                              # (num_anchors, 4+nc)
        boxes_cxcywh = pred[:, :4]
        class_probs  = pred[:, 4:]

        anchor_ids, cls_ids = (class_probs > conf_threshold).nonzero(as_tuple=True)

        if anchor_ids.numel() == 0:
            results.append({
                'boxes': boxes_cxcywh.new_zeros((0, 4)),
                'scores': boxes_cxcywh.new_zeros((0,)),
                'labels': cls_ids.new_zeros((0,), dtype=torch.long),
            })
            continue

        boxes_cxcywh = boxes_cxcywh[anchor_ids]
        conf         = class_probs[anchor_ids, cls_ids]
        order        = conf.argsort(descending=True)[:max_nms]
        boxes_cxcywh = boxes_cxcywh[order]
        conf         = conf[order]
        cls_ids      = cls_ids[order]

        boxes_xyxy = cxcywh_to_xyxy(boxes_cxcywh)
        keep       = batched_nms(boxes_xyxy, conf, cls_ids, iou_threshold)[:max_det]

        results.append({
            'boxes':  boxes_xyxy[keep],
            'scores': conf[keep],
            'labels': cls_ids[keep],
        })
    return results


def build_gt(targets, batch_size, img_size):
    """Convert the flat targets dict back to per-image format for torchmetrics."""
    gt = []
    for i in range(batch_size):
        mask        = targets['idx'] == i
        boxes_norm  = targets['box'][mask]                        # (N,4) normalised
        boxes_px    = boxes_norm.clone()
        boxes_px[:, [0, 2]] *= img_size
        boxes_px[:, [1, 3]] *= img_size
        gt.append({
            'boxes':  cxcywh_to_xyxy(boxes_px),
            'labels': targets['cls'][mask].squeeze(1).long(),
        })
    return gt


def box_iou_xyxy(boxes1, boxes2):
    """Pairwise IoU for xyxy boxes."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)

    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-9)


def init_class_stats():
    return {cls_id: {'tp': 0, 'fp': 0, 'fn': 0} for cls_id in CLASS_NAMES}


def update_class_stats(stats, preds, gt, iou_thr=0.50):
    """
    Greedy one-to-one matching for class-level precision/recall/F1.
    Predictions are matched to ground truth of the same class only.
    """
    for pred, target in zip(preds, gt):
        for cls_id in CLASS_NAMES:
            pred_mask = pred['labels'] == cls_id
            gt_mask = target['labels'] == cls_id

            pred_boxes = pred['boxes'][pred_mask]
            pred_scores = pred['scores'][pred_mask]
            gt_boxes = target['boxes'][gt_mask]

            if pred_boxes.numel() == 0:
                stats[cls_id]['fn'] += int(gt_boxes.shape[0])
                continue
            if gt_boxes.numel() == 0:
                stats[cls_id]['fp'] += int(pred_boxes.shape[0])
                continue

            order = pred_scores.argsort(descending=True)
            ious = box_iou_xyxy(pred_boxes, gt_boxes)
            matched_gt = torch.zeros(gt_boxes.shape[0], dtype=torch.bool, device=gt_boxes.device)

            for pred_idx in order:
                best_iou, best_gt_idx = ious[pred_idx].max(dim=0)
                if best_iou >= iou_thr and not matched_gt[best_gt_idx]:
                    stats[cls_id]['tp'] += 1
                    matched_gt[best_gt_idx] = True
                else:
                    stats[cls_id]['fp'] += 1

            stats[cls_id]['fn'] += int((~matched_gt).sum().item())


def _pr_f1(tp, fp, fn):
    """Precision/recall/F1 from raw counts — shared by per-class and overall."""
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1


def finalize_class_stats(stats):
    final = {}
    for cls_id, counts in stats.items():
        tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
        precision, recall, f1 = _pr_f1(tp, fp, fn)
        final[cls_id] = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    return final


def compute_overall_stats(class_stats):
    """
    Micro-averaged precision/recall/F1 across all classes: pool TP/FP/FN over
    every class, then compute the three metrics once. This is the dataset-level
    number (not per-class) — dominated by nonpollenbee since it's the majority.
    """
    tp = sum(c['tp'] for c in class_stats.values())
    fp = sum(c['fp'] for c in class_stats.values())
    fn = sum(c['fn'] for c in class_stats.values())
    precision, recall, f1 = _pr_f1(tp, fp, fn)
    return {'tp': tp, 'fp': fp, 'fn': fn,
            'precision': precision, 'recall': recall, 'f1': f1}


def _per_class_result(res, key='map_per_class', classes_key='classes'):
    values = res[key]
    if values.ndim == 0:
        values = values.reshape(1)

    classes = res.get(classes_key)
    if classes is None:
        classes = torch.arange(values.numel(), device=values.device)
    elif classes.ndim == 0:
        classes = classes.reshape(1)

    if classes.numel() != values.numel():
        n = min(classes.numel(), values.numel())
        classes = classes[:n]
        values = values[:n]

    return classes, values


def pollenbee_fitness(res, pollen_id):
    """YOLO-style 'fitness' restricted to the pollenbee (minority) class:
        fitness = 0.1 * AP@0.5 + 0.9 * AP@0.5:0.95   (pollenbee only)
    This is the Ultralytics fitness blend but computed on the pollenbee
    per-class AP instead of the all-class mAP, so it keeps the minority-class
    focus best.pt is selected for while being smoother than the raw pollenbee F1
    (which sits at exactly 0 for many early epochs). torchmetrics reports -1 for
    an absent class, so each term is floored at 0; returns 0.0 until pollenbee
    starts getting detected.
    """
    def _ap(per_class_key, classes_key):
        classes, values = _per_class_result(res, key=per_class_key, classes_key=classes_key)
        for cid, v in zip(classes.tolist(), values.tolist()):
            if int(cid) == pollen_id:
                return max(float(v), 0.0)
        return 0.0
    ap50    = _ap('map_50_per_class', 'classes_50')
    ap50_95 = _ap('map_per_class', 'classes')
    return 0.1 * ap50 + 0.9 * ap50_95


def metric_to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def metrics_csv_fields():
    fields = [
        'run_id',
        'model_name',
        'epoch',
        'lr',
        'train_loss',
        'map_50',
        'map_50_95',
        'stats_conf_thr',
        'stats_iou_thr',
        'overall_precision',
        'overall_recall',
        'overall_f1',
        'overall_tp',
        'overall_fp',
        'overall_fn',
    ]
    for cls_id in sorted(CLASS_NAMES):
        name = CLASS_NAMES[cls_id]
        fields.extend([
            f'{name}_precision',
            f'{name}_recall',
            f'{name}_f1',
            f'{name}_tp',
            f'{name}_fp',
            f'{name}_fn',
        ])
    return fields


def build_metrics_row(run_id, model_name, epoch, lr, train_loss, eval_res, stats_conf_thr, stats_iou_thr):
    row = {
        'run_id': run_id,
        'model_name': model_name,
        'epoch': epoch,
        'lr': lr,
        'train_loss': train_loss,
        'map_50': metric_to_float(eval_res['map_50']),
        'map_50_95': metric_to_float(eval_res['map']),
        'stats_conf_thr': stats_conf_thr,
        'stats_iou_thr': stats_iou_thr,
    }

    overall = eval_res['overall_stats']
    row['overall_precision'] = overall['precision']
    row['overall_recall'] = overall['recall']
    row['overall_f1'] = overall['f1']
    row['overall_tp'] = overall['tp']
    row['overall_fp'] = overall['fp']
    row['overall_fn'] = overall['fn']

    for cls_id in sorted(CLASS_NAMES):
        name = CLASS_NAMES[cls_id]
        cls_stats = eval_res['class_stats'][cls_id]
        row[f'{name}_precision'] = cls_stats['precision']
        row[f'{name}_recall'] = cls_stats['recall']
        row[f'{name}_f1'] = cls_stats['f1']
        row[f'{name}_tp'] = cls_stats['tp']
        row[f'{name}_fp'] = cls_stats['fp']
        row[f'{name}_fn'] = cls_stats['fn']

    return row


def append_metrics_csv(csv_path, row, fieldnames):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def evaluate(model, loader, device, img_size=640, conf_thr=0.001, iou_thr=0.70, stats_conf_thr=0.25, stats_iou_thr=0.50):
    """
    Evaluate on a data split.  Returns a dict containing at minimum:
        map_50   — mAP at IoU=0.50  (primary metric for the paper)
        map      — mAP at IoU=0.50:0.95
    """
    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
    except ImportError:
        raise ImportError('Run: pip install torchmetrics')

    model.eval()
    # NOTE: do NOT override max_detection_thresholds. torchmetrics only computes
    # the summary `map` (mAP@0.50:0.95) when 100 is one of the thresholds; passing
    # [1, 10, 300] silently returns map = -1. The default [1, 10, 100] is COCO-
    # standard and the densest image here has 84 boxes, so the 100 cap loses nothing.
    metric = MeanAveragePrecision(
        iou_thresholds=[x / 100 for x in range(50, 100, 5)],
        class_metrics=True,
    )
    metric50 = MeanAveragePrecision(
        iou_thresholds=[0.5],
        class_metrics=True,
    )
    for m in (metric, metric50):
        if hasattr(m, 'warn_on_many_detections'):
            m.warn_on_many_detections = False

    class_stats = init_class_stats()

    with torch.no_grad():
        for imgs, targets in loader:
            imgs           = imgs.to(device)
            targets['box'] = targets['box'].to(device)
            targets['cls'] = targets['cls'].to(device)
            targets['idx'] = targets['idx'].to(device)

            raw = model(imgs)                             # (B, 4+nc, anchors)
            preds_per_img = [raw[i] for i in range(imgs.shape[0])]
            nms_out       = non_max_suppression(preds_per_img, conf_thr, iou_thr)
            stats_out     = non_max_suppression(preds_per_img, stats_conf_thr, iou_thr)
            gt            = build_gt(targets, imgs.shape[0], img_size)

            preds_cpu = [{k: v.cpu() for k, v in p.items()} for p in nms_out]
            gt_cpu = [{k: v.cpu() for k, v in g.items()} for g in gt]
            metric.update(preds_cpu, gt_cpu)
            metric50.update(preds_cpu, gt_cpu)
            update_class_stats(class_stats, stats_out, gt, iou_thr=stats_iou_thr)

    res = metric.compute()
    res50 = metric50.compute()
    classes50, map50_per_class = _per_class_result(res50)
    classes, map_per_class = _per_class_result(res)
    res['map_50_per_class'] = map50_per_class
    res['classes_50'] = classes50
    res['class_stats'] = finalize_class_stats(class_stats)
    res['overall_stats'] = compute_overall_stats(res['class_stats'])

    log.info(f'  mAP@0.50      : {res["map_50"]:.4f}')
    log.info(f'  mAP@0.50:0.95 : {res["map"]:.4f}')
    for cid, ap in zip(res['classes_50'].tolist(), res['map_50_per_class']):
        log.info(f'  AP@0.50 [{CLASS_NAMES.get(cid, cid)}]: {ap:.4f}')
    for cid, ap in zip(classes.tolist(), map_per_class):
        log.info(f'  AP@0.50:0.95 [{CLASS_NAMES.get(cid, cid)}]: {ap:.4f}')
    for cls_id, cls_stats in res['class_stats'].items():
        log.info(
            f'  Stats@conf{stats_conf_thr:.3f}/IoU{stats_iou_thr:.2f} '
            f'[{CLASS_NAMES.get(cls_id, cls_id)}]: '
            f'precision={cls_stats["precision"]:.4f}  '
            f'recall={cls_stats["recall"]:.4f}  '
            f'F1={cls_stats["f1"]:.4f}'
        )
    ov = res['overall_stats']
    log.info(
        f'  Stats@conf{stats_conf_thr:.3f}/IoU{stats_iou_thr:.2f} '
        f'[overall]: precision={ov["precision"]:.4f}  '
        f'recall={ov["recall"]:.4f}  F1={ov["f1"]:.4f}'
    )
    return res


# ===========================================================================
# PART 7 — Main
# ===========================================================================

def main():
    # ── Paths — edit DATASET_ROOT to point at your pollenbees/ folder ────
    # POLLENBEES_DATA overrides the dataset root (set it on Colab, e.g.
    # /content/pollenbees). Falls back to the local Windows path.
    DATASET_ROOT = os.environ.get('POLLENBEES_DATA', r'E:\Alireza\pollenbees')
    TRAIN_ROOT   = os.path.join(DATASET_ROOT, 'train')
    VAL_ROOT     = os.path.join(DATASET_ROOT, 'val')
    TEST_ROOT    = os.path.join(DATASET_ROOT, 'test')

    # ── Hyperparameters (standard YOLOv8 defaults — cite the original paper)
    IMG_SIZE     = 1280 #changed from 640 
    BATCH_SIZE   = 16
    NUM_EPOCHS   = int(os.environ.get('POLLENBEES_EPOCHS', 100))
    LR           = 0.01          # initial learning rate
    MOMENTUM     = 0.937
    WEIGHT_DECAY = 5e-4
    SEED         = 42
    ATTN         = os.environ.get('POLLENBEES_ATTN', 'none')   # none|cbam|botnet|cbam_botnet
    _default_name = 'custom_yolov8s_baseline' if ATTN == 'none' else f'custom_yolov8s_{ATTN}'
    MODEL_NAME   = os.environ.get('POLLENBEES_MODEL_NAME', _default_name)
    RUN_ID       = os.environ.get('POLLENBEES_RUN_ID', datetime.now().strftime('%Y%m%d_%H%M%S'))
    EVAL_EVERY   = 1
    STATS_CONF_THR = 0.5
    STATS_IOU_THR  = 0.50
    # Early stopping on pollenbee-weighted fitness (0.1*AP@0.5 + 0.9*AP@0.5:0.95
    # over the pollenbee class), the same metric best.pt is selected by.
    # PATIENCE=0 disables it. MIN_EPOCHS is a floor: the ~3% pollenbee class often
    # sits at fitness=0 for many early epochs before it starts being detected, so
    # we never stop before then or we'd kill the run during that plateau.
    EARLY_STOP_PATIENCE   = int(os.environ.get('POLLENBEES_PATIENCE', 20))
    EARLY_STOP_MIN_EPOCHS = int(os.environ.get('POLLENBEES_MIN_EPOCHS', 30))
    # POLLENBEES_OUT sends checkpoints/CSVs somewhere persistent (e.g. a Drive
    # path) while data can live on fast local disk. Defaults to DATASET_ROOT.
    OUT_ROOT     = os.environ.get('POLLENBEES_OUT', DATASET_ROOT)
    SAVE_DIR     = os.path.join(OUT_ROOT, 'runs', 'baseline')
    RESULTS_DIR  = os.path.join(OUT_ROOT, 'runs', 'results')
    RESULTS_CSV  = os.path.join(RESULTS_DIR, f'{MODEL_NAME}_{RUN_ID}.csv')
    DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Reproducibility ───────────────────────────────────────────────────
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    metric_fields = metrics_csv_fields()

    log.info(f'Device: {DEVICE}')
    log.info(f'Dataset root: {DATASET_ROOT}')
    log.info(f'Model name: {MODEL_NAME}')
    log.info(f'Run ID: {RUN_ID}')
    log.info(f'Metrics CSV: {RESULTS_CSV}')

    # ── Step 1: Convert annotations (skipped automatically if already done)
    for split_root in [TRAIN_ROOT, VAL_ROOT, TEST_ROOT]:
        convert_split(split_root)

    # ── Step 2: Build datasets ────────────────────────────────────────────
    log.info('--- Train split ---')
    train_ds = PollenBeeDataset(TRAIN_ROOT, img_size=IMG_SIZE)

    log.info('--- Val split ---')
    val_ds   = PollenBeeDataset(VAL_ROOT,   img_size=IMG_SIZE)

    # num_workers via env (POLLENBEES_WORKERS). persistent_workers reuses the
    # workers across epochs instead of re-spawning them every epoch — on Windows
    # the per-epoch spawn/teardown churn eventually makes workers "exit
    # unexpectedly" during a long run. Set POLLENBEES_WORKERS=0 for a worker-free
    # (slower but bulletproof) run.
    NUM_WORKERS = int(os.environ.get('POLLENBEES_WORKERS', 4))
    # pin_memory defaults OFF on Windows: with num_workers>0 the pin-memory
    # thread calls cudaHostRegister on the workers' shared-memory batches and
    # can hit "CUDA error: resource already mapped" (cudaErrorAlreadyMapped),
    # which crashes the run before epoch 1. Large 1280px batches make it worse.
    # Set POLLENBEES_PIN=1 to re-enable if your setup tolerates it.
    PIN_MEMORY = DEVICE == 'cuda' and os.environ.get('POLLENBEES_PIN', '0') == '1'
    log.info(f'DataLoader workers: {NUM_WORKERS}  pin_memory: {PIN_MEMORY}')
    loader_kwargs = dict(
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY, collate_fn=collate_fn,
        persistent_workers=NUM_WORKERS > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    # ── Step 3: Model ─────────────────────────────────────────────────────
    # Import MyYolo from the file where you saved your notebook code.
    # Make sure Head accepts num_classes=2 (not hardcoded to 80).
    from yolov8_model import MyYolo
    import yaml
    from utils import util

    log.info(f'Attention mode: {ATTN}')
    model = MyYolo(version='s', num_classes=2, attn=ATTN, img_size=IMG_SIZE).to(DEVICE)
    log.info(f'Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

    with open('utils/args.yaml') as f:
        params = yaml.safe_load(f)
    # Inject per-class GT counts from the TRAIN split so the loss can build
    # effective-number class weights (only used when cls_effective_number is on).
    params['class_counts'] = [train_ds.class_counts.get(i, 0) for i in range(len(CLASS_NAMES))]
    log.info(f'Train class counts (for class weighting): {params["class_counts"]}')
    criterion = util.ComputeLoss(model, params)

    # ── Step 4: Optimiser + scheduler ────────────────────────────────────
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LR,
        momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True,
    )
    # Cosine annealing: decays LR from LR → LR*0.01 over NUM_EPOCHS.
    # Standard for YOLO-family — reviewers will recognise this.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=LR * 0.01,
    )

    # ── Step 5: Train ────────────────────────────────────────────────────
    # Select best.pt by pollenbee-weighted fitness (0.1*AP@0.5 + 0.9*AP@0.5:0.95
    # on the pollenbee class), not overall mAP@0.50. Pollenbee is the ~3% minority
    # class that matters for this dataset; picking on all-class mAP@0.50 rewards
    # the majority class and can save a checkpoint that barely detects pollenbee
    # (as happened in an earlier run — best mAP epoch had pollenbee F1 ~0.5). The
    # fitness blend keeps that minority-class focus but is smoother than raw F1.
    POLLENBEE_ID = CLASS_MAP['pollenbee']
    history        = {'train_loss': [], 'val_metrics': []}
    best_fitness   = -1.0
    epochs_no_improve = 0   # for early stopping

    for epoch in range(1, NUM_EPOCHS + 1):
        current_lr = scheduler.get_last_lr()[0]
        log.info(f'=== Epoch {epoch}/{NUM_EPOCHS}  LR={current_lr:.6f} ===')

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, DEVICE, epoch
        )
        history['train_loss'].append(train_loss)
        log.info(f'Epoch {epoch} avg train loss: {train_loss:.4f}')

        # Evaluate every epoch so each model run has a complete metrics CSV.
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            log.info('--- Validation ---')
            res = evaluate(
                model, val_loader, DEVICE, img_size=IMG_SIZE,
                stats_conf_thr=STATS_CONF_THR, stats_iou_thr=STATS_IOU_THR,
            )
            map50 = metric_to_float(res['map_50'])
            pollen_f1 = res['class_stats'][POLLENBEE_ID]['f1']
            fitness   = pollenbee_fitness(res, POLLENBEE_ID)
            metrics_row = build_metrics_row(
                RUN_ID, MODEL_NAME, epoch, current_lr, train_loss,
                res, STATS_CONF_THR, STATS_IOU_THR,
            )
            append_metrics_csv(RESULTS_CSV, metrics_row, metric_fields)
            history['val_metrics'].append(metrics_row)
            log.info(f'Epoch {epoch} metrics appended → {RESULTS_CSV}')
            log.info(f'Epoch {epoch} pollenbee fitness: {fitness:.4f}')

            if fitness > best_fitness:
                best_fitness = fitness
                epochs_no_improve = 0
                ckpt = os.path.join(SAVE_DIR, 'best.pt')
                torch.save({
                    'epoch':        epoch,
                    'model':        model.state_dict(),
                    'optimizer':    optimizer.state_dict(),
                    'box_loss':     criterion.box_loss.state_dict(),  # WIoU running_mean + class_weight
                    'map50':        map50,
                    'pollenbee_f1': pollen_f1,
                    'fitness':      fitness,
                    'config': {
                        'model_name': MODEL_NAME,
                        'run_id': RUN_ID,
                        'version': 's',
                        'img_size': IMG_SIZE,
                        'num_classes': 2,
                    },
                }, ckpt)
                log.info(
                    f'New best saved → {ckpt}  (fitness={fitness:.4f}, '
                    f'pollenbee F1={pollen_f1:.4f}, mAP@0.50={map50:.4f})'
                )
            else:
                epochs_no_improve += 1
                log.info(
                    f'No pollenbee-fitness improvement for {epochs_no_improve}/{EARLY_STOP_PATIENCE} '
                    f'epoch(s) (best={best_fitness:.4f})'
                )

            # Early stopping. Counts in eval-events; with EVAL_EVERY=1 that is epochs.
            # best.pt already holds the best epoch, so stopping only saves compute —
            # it never costs you the best model.
            if (EARLY_STOP_PATIENCE > 0
                    and epoch >= EARLY_STOP_MIN_EPOCHS
                    and epochs_no_improve >= EARLY_STOP_PATIENCE):
                log.info(
                    f'Early stopping at epoch {epoch}: no pollenbee-fitness improvement for '
                    f'{epochs_no_improve} epochs (best={best_fitness:.4f}).'
                )
                break

    log.info(f'Training complete.  Best pollenbee fitness: {best_fitness:.4f}')

    import json as _json
    with open(os.path.join(SAVE_DIR, 'history.json'), 'w') as f:
        _json.dump(history, f, indent=2)

    # ── Step 6: Final evaluation on the held-out TEST split ───────────────
    # Load best.pt (selected by val pollenbee F1) so we report the chosen
    # checkpoint, not whatever the last epoch happened to be.
    best_ckpt = os.path.join(SAVE_DIR, 'best.pt')
    if os.path.isdir(annotations_dir(TEST_ROOT)) and os.path.exists(best_ckpt):
        log.info('--- Test split (final, using best.pt) ---')
        model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE)['model'])
        test_ds     = PollenBeeDataset(TEST_ROOT, img_size=IMG_SIZE)
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=4, pin_memory=PIN_MEMORY, collate_fn=collate_fn,
        )
        test_res = evaluate(
            model, test_loader, DEVICE, img_size=IMG_SIZE,
            stats_conf_thr=STATS_CONF_THR, stats_iou_thr=STATS_IOU_THR,
        )
        test_row = build_metrics_row(
            RUN_ID, MODEL_NAME, 'test', '', '',
            test_res, STATS_CONF_THR, STATS_IOU_THR,
        )
        append_metrics_csv(RESULTS_CSV, test_row, metric_fields)
        log.info(f'Test metrics appended (epoch="test") → {RESULTS_CSV}')
    else:
        log.warning('Skipping test eval: test annotations or best.pt not found.')


if __name__ == '__main__':
    main()
