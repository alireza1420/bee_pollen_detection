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

import os, json, glob, random, logging
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


def evaluate(model, loader, device, img_size=640, conf_thr=0.001, iou_thr=0.70):
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
    metric = MeanAveragePrecision(
        iou_thresholds=[x / 100 for x in range(50, 100, 5)],
        max_detection_thresholds=[1, 10, 300],
        class_metrics=True,
    )
    metric50 = MeanAveragePrecision(
        iou_thresholds=[0.5],
        max_detection_thresholds=[1, 10, 300],
        class_metrics=True,
    )
    for m in (metric, metric50):
        if hasattr(m, 'warn_on_many_detections'):
            m.warn_on_many_detections = False

    with torch.no_grad():
        for imgs, targets in loader:
            imgs           = imgs.to(device)
            targets['box'] = targets['box'].to(device)
            targets['cls'] = targets['cls'].to(device)
            targets['idx'] = targets['idx'].to(device)

            raw = model(imgs)                             # (B, 4+nc, anchors)
            preds_per_img = [raw[i] for i in range(imgs.shape[0])]
            nms_out       = non_max_suppression(preds_per_img, conf_thr, iou_thr)
            gt            = build_gt(targets, imgs.shape[0], img_size)

            preds_cpu = [{k: v.cpu() for k, v in p.items()} for p in nms_out]
            gt_cpu = [{k: v.cpu() for k, v in g.items()} for g in gt]
            metric.update(preds_cpu, gt_cpu)
            metric50.update(preds_cpu, gt_cpu)

    res = metric.compute()
    res50 = metric50.compute()
    classes50, map50_per_class = _per_class_result(res50)
    classes, map_per_class = _per_class_result(res)
    res['map_50_per_class'] = map50_per_class
    res['classes_50'] = classes50

    log.info(f'  mAP@0.50      : {res["map_50"]:.4f}')
    log.info(f'  mAP@0.50:0.95 : {res["map"]:.4f}')
    for cid, ap in zip(res['classes_50'].tolist(), res['map_50_per_class']):
        log.info(f'  AP@0.50 [{CLASS_NAMES.get(cid, cid)}]: {ap:.4f}')
    for cid, ap in zip(classes.tolist(), map_per_class):
        log.info(f'  AP@0.50:0.95 [{CLASS_NAMES.get(cid, cid)}]: {ap:.4f}')
    return res


# ===========================================================================
# PART 7 — Main
# ===========================================================================

def main():
    # ── Paths — edit DATASET_ROOT to point at your pollenbees/ folder ────
    DATASET_ROOT = r'E:\Alireza\pollenbees'    # Windows path — use raw string
    TRAIN_ROOT   = os.path.join(DATASET_ROOT, 'train')
    VAL_ROOT     = os.path.join(DATASET_ROOT, 'val')
    TEST_ROOT    = os.path.join(DATASET_ROOT, 'test')

    # ── Hyperparameters (standard YOLOv8 defaults — cite the original paper)
    IMG_SIZE     = 640
    BATCH_SIZE   = 16
    NUM_EPOCHS   = 100
    LR           = 0.01          # initial learning rate
    MOMENTUM     = 0.937
    WEIGHT_DECAY = 5e-4
    SEED         = 42
    SAVE_DIR     = os.path.join(DATASET_ROOT, 'runs', 'baseline')
    DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Reproducibility ───────────────────────────────────────────────────
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    log.info(f'Device: {DEVICE}')
    log.info(f'Dataset root: {DATASET_ROOT}')

    # ── Step 1: Convert annotations (skipped automatically if already done)
    for split_root in [TRAIN_ROOT, VAL_ROOT, TEST_ROOT]:
        convert_split(split_root)

    # ── Step 2: Build datasets ────────────────────────────────────────────
    log.info('--- Train split ---')
    train_ds = PollenBeeDataset(TRAIN_ROOT, img_size=IMG_SIZE)

    log.info('--- Val split ---')
    val_ds   = PollenBeeDataset(VAL_ROOT,   img_size=IMG_SIZE)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_fn,
    )

    # ── Step 3: Model ─────────────────────────────────────────────────────
    # Import MyYolo from the file where you saved your notebook code.
    # Make sure Head accepts num_classes=2 (not hardcoded to 80).
    from yolov8_model import MyYolo
    import yaml
    from utils import util

    model = MyYolo(version='s', num_classes=2).to(DEVICE)
    log.info(f'Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

    with open('utils/args.yaml') as f:
        params = yaml.safe_load(f)
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
    history    = {'train_loss': [], 'val_map50': []}
    best_map50 = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        current_lr = scheduler.get_last_lr()[0]
        log.info(f'=== Epoch {epoch}/{NUM_EPOCHS}  LR={current_lr:.6f} ===')

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, DEVICE, epoch
        )
        history['train_loss'].append(train_loss)
        log.info(f'Epoch {epoch} avg train loss: {train_loss:.4f}')

        # Evaluate every 5 epochs and at the final epoch.
        # mAP is expensive to compute — don't do it every epoch.
        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            log.info('--- Validation ---')
            res   = evaluate(model, val_loader, DEVICE, img_size=IMG_SIZE)
            map50 = res['map_50'].item()
            history['val_map50'].append({'epoch': epoch, 'map50': map50})

            if map50 > best_map50:
                best_map50 = map50
                ckpt = os.path.join(SAVE_DIR, 'best.pt')
                torch.save({
                    'epoch':      epoch,
                    'model':      model.state_dict(),
                    'optimizer':  optimizer.state_dict(),
                    'map50':      map50,
                    'config':     {'version': 's', 'img_size': IMG_SIZE, 'num_classes': 2},
                }, ckpt)
                log.info(f'New best saved → {ckpt}  (mAP@0.50={map50:.4f})')

    log.info(f'Training complete.  Best mAP@0.50: {best_map50:.4f}')

    import json as _json
    with open(os.path.join(SAVE_DIR, 'history.json'), 'w') as f:
        _json.dump(history, f, indent=2)


if __name__ == '__main__':
    main()
