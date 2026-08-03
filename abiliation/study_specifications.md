# Ablation protocol — checklist sheet

Project: pollen-bearing bee detection (VnPollenBee) · YOLOv8 + CBAM + BoT + SAF-SIoU

---

## 0. Pre-flight — run this FIRST

- [ ] **Run `s@1280`, full 100-epoch schedule, early stopping OFF.**
      This decides which backbone the paper is about. Current evidence: `s@1280/38ep`
      reached pollen F1 **0.929** vs `m@1280/94ep` at **0.927** — the small model
      already matches on the benchmark metric, and its apparent mAP deficit is
      confounded with the truncated schedule.
- [ ] Compare against `m@1280` on **pollen AP@0.5:0.95** (validation).
- [ ] Benchmark FPS for both `s` and `m` (§5). If accuracy is within noise, speed
      decides — and converts "no worse" into a deployment argument.
- [x] **Decide the base model** and record the decision here: **YOLOv8m**
      (decided 2026-07-27 — the ablation runs on `m` only, no `s` variant)
      → If `s` is within noise of `m`, use `s`: cheaper ablation, faster inference,
        stronger deployment story, and ablation matches the reported model.
- [ ] Confirm split provenance: are you using the **official** VnPollenBee split?
      (Box counts match published totals exactly — verify the loader.)
      → Yes / No: ................

---

## 1. Invariants — identical across EVERY run

Any variation here invalidates the table.

- [ ] Data split (train/val/test) — never re-shuffled
- [ ] Input resolution — 1280×1280
- [ ] Epochs — same fixed count for all ablation rows (100, or 70 if budget-bound)
- [ ] **Early stopping OFF for all ablation runs**
- [ ] Optimizer, LR schedule (cosine), warmup, batch size
- [ ] Augmentation ON for every run, train split only — HSV jitter
      (0.015/0.7/0.4) + horizontal flip 0.5. Mosaic/affine OFF for all rows
      (scale-preserving: object-size statistics stay matched to evaluation)
- [ ] Seed — single seed **42** for every row (configurable per run via
      `POLLENBEES_SEED`, recorded in the CSV `seed` column)
- [ ] Checkpoint-selection criterion — pollen-weighted fitness, same for all rows
- [ ] Evaluation protocol — validation set, conf 0.5, IoU 0.5

---

## 2. Component definitions — what "off" means

| Component | ON | OFF |
|---|---|---|
| CBAM | CBAM on P3–P5 backbone maps | modules removed entirely |
| BoT | C2f of final backbone stage → BoT hybrid bottlenecks (MHSA + rel. pos.) | standard C2f retained |
| SAF-SIoU | `saf_*` composite loss (SIoU + NWD gate + WIoU-v3 focusing) | stock **CIoU** |
| VFL-w | VFL with `cls_pos_weight = [1.0, 2.0]` | stock **BCE** (`cls_loss_type: bce`) |

> Decision 2026-07-27: the classification component is ablated as one bundle —
> weighted VFL (ON) vs stock BCE (OFF), so the baseline is stock YOLOv8.
> Row 4 therefore measures the combined effect of Varifocal loss + minority
> weighting, not the weighting alone.

---

## 3. Component ablation matrix

Base model: **YOLOv8m** (§0) · Resolution 1280 · Fixed epochs · No early stopping

| # | Run name | CBAM | BoT | SAF-SIoU | VFL-w | Purpose | Done |
|---|---|:---:|:---:|:---:|:---:|---|:---:|
| 0 | `abl_baseline` | – | – | – | – | Stock YOLOv8 baseline | ☐ |
| 1 | `abl_cbam` | ✓ | – | – | – | CBAM alone | ☐ |
| 2 | `abl_bot` | – | ✓ | – | – | BoT alone | ☐ |
| 3 | `abl_saf` | – | – | ✓ | – | SAF-SIoU alone | ☐ |
| 4 | `abl_vflw` | – | – | – | ✓ | Minority weighting alone | ☐ |
| 5 | `abl_no_cbam` | – | ✓ | ✓ | ✓ | Full − CBAM | ☐ |
| 6 | `abl_no_bot` | ✓ | – | ✓ | ✓ | Full − BoT | ☐ |
| 7 | `abl_no_saf` | ✓ | ✓ | – | ✓ | Full − SAF-SIoU | ☐ |
| 8 | `abl_no_vflw` | ✓ | ✓ | ✓ | – | Full − weighting | ☐ |
| 9 | `abl_full` | ✓ | ✓ | ✓ | ✓ | **Full model** | ☐ |
| 10 | `abl_cbam_saf` | ✓ | – | ✓ | – | **Candidate final model** (post-hoc) | ☐ |

**Minimum publishable subset if compute-bound:** rows 0–4 and 9 (six runs).
Rows 5–8 (leave-one-out) reveal redundancy that leave-one-in cannot — e.g. a module
that helps alone but adds nothing on top of the others.

**Row 10 is a post-hoc addition (2026-08-03), not part of the pre-registered
factorial design.** It is a 2-component combination, so the one-factor rule (§8.1)
does not apply to it — it is a *proposed final model*, not an ablation contrast.
Motivation from rows 0–9: the pre-registered full model (row 9) *regressed* vs
baseline; the leave-one-out rows localized the cause to **BoT** (row 6, Full−BoT,
raised val AP +0.024 — the largest recovery) and to **VFL-w**'s conf-0.5 precision
collapse (row 6: P 0.811, FAR 0.189). `abl_cbam_saf` keeps only the two
precision-clean components (CBAM row 1, SAF row 3) and drops both suspects. Report
it as a model selected from ablation evidence; **rank on validation and keep the
test set sealed** until one final model is committed (§8.2) — do not test-evaluate
every candidate subset (multiple-comparison leakage).

---

## 4. Resolution evidence — from existing runs, no new compute

Not a sub-ablation. Contribution bullet #2 ("input resolution is the dominant
factor") depends on this comparison, so it must appear somewhere in the paper —
but the runs are already done. Report it as a short table or as narrative in the
results section.

| Source run | imgsz | pollen F1 (test) | mAP@0.5:0.95 (test) |
|---|---|---|---|
| `s@640, w2` | 640 | 0.897 | 0.731 |
| `s@1280` | 1280 | 0.929 | 0.731 |
| `m@1280, full schedule` | 1280 | 0.927 | 0.782 |

- [ ] State clearly that these differ in training length, so they are presented as
      supporting evidence rather than a controlled ablation.

---

## 5. Metrics to log for every run

Primary (ablation ranking):

- [ ] **pollen AP@0.5:0.95** ← primary. Threshold-free, avoids the F1@0.5 trap
- [ ] pollen AP@0.5

Secondary (comparability with Nguyen et al.):

- [ ] pollen P / R / F1 @ conf 0.5
- [ ] pollen MR = FN/(FN+TP), FAR = FP/(TP+FP)
- [ ] pollen TP / FP / FN (raw counts)

Guardrails:

- [ ] overall mAP@0.5, mAP@0.5:0.95
- [ ] non-pollen F1 (check the majority class doesn't regress)

Bookkeeping:

- [ ] best epoch, total epochs, wall-clock, GPU

### Inference speed — required, not optional

Three claims depend on it: the abstract's real-time framing, the "one-stage
suited to continuous monitoring" contribution, and the differentiation from
Nguyen et al.'s two-stage method. Shi et al. report 51 fps / 31.8 GFLOPs;
Sledevič reports 8.8 ms/frame and 36 fps. Omission will be noticed.

- [ ] Benchmark **both `s` and `m`** at 1280 (feeds the §0 base-model decision)
- [ ] Batch size 1 — the realistic streaming case
- [ ] 10–20 warmup iterations before timing
- [ ] `torch.cuda.synchronize()` around the timed region
- [ ] Report preprocess / inference / postprocess ms, total ms, FPS — mean and std
- [ ] Run in **both FP32 and FP16**, report both
- [ ] State GPU model, VRAM, torch/CUDA versions, precision
- [ ] Optional: TensorRT/ONNX export FPS, clearly labelled as an optimised export
      separate from the native PyTorch figure
- [ ] Frame as **per-frame detection throughput**, not end-to-end video
      processing — VnPollenBee ships images, not video
- [ ] Decide and state the real-time threshold you are claiming against

---

## 6. Version control — one branch per run

**No run starts from a dirty or shared working tree.** Every row in §3 gets its
own branch, so the exact code that produced each CSV is recoverable months later
when a reviewer asks.

Per-run procedure:

- [ ] Start from a clean, up-to-date `main`: `git status` must be empty
- [ ] `git checkout -b run/<run_name>` — name matches §3 exactly
      (e.g. `run/abl_cbam`, `run/abl_no_bot`, `run/abl_full`)
- [ ] Apply **only** the config/code changes for that one row
- [ ] Commit **before** training starts — the code state must predate the results
- [ ] `git push -u origin run/<run_name>`
- [ ] Record the commit SHA in the results log next to the run's metrics
- [ ] After the run, commit the outputs on the same branch:
      results CSV, `args.yaml`, best-checkpoint metadata, seed
- [ ] Never amend or force-push a branch whose results are already recorded

Rules:

1. **One branch = one row = one config delta.** If a branch contains two changes,
   the row is invalid.
2. **Branch per seed** if seeds require code/config changes:
   `run/abl_full_s0`, `run/abl_full_s1`, … Otherwise pass the seed as an argument
   and record it in the log.
3. Merge nothing back into `main` mid-study — `main` stays the fixed reference
   for all rows.
4. If a run crashes and is restarted with changed settings, that is a **new
   branch**, not a continuation.
5. Tag the final model's branch (e.g. `git tag paper-final`) once the test-set
   evaluation is done.

Run log columns to maintain: `run_name · branch · commit SHA · seed · epochs ·
best epoch · pollen AP@0.5:0.95 · pollen F1 · notes`

---

## 7. Statistical rules

- [ ] **2–3 seeds per configuration**, report mean ± std.
      Validation has 307 pollen instances; differences of 0.01–0.02 F1 are noise.
- [ ] If seeds are unaffordable: run 3 seeds for rows 0 and 9 only, use that spread
      as a **noise floor**, and mark any row inside it as not significant.
- [ ] State the noise floor explicitly in the caption.
- [ ] Never claim an improvement smaller than the seed spread.

---

## 8. Hard rules

1. **One factor per row.** Never change two things in a run.
2. **Ablate on validation. Test set stays sealed** until the final model.
3. **Rank by AP, not F1@0.5.** A fixed threshold once made an improved detector
   look worse (F1 0.929 → 0.899 while mAP@0.5:0.95 rose 0.731 → 0.753).
4. **No early stopping in ablation runs** — variable duration is a confound.
5. **Same epoch count everywhere**, including the config study.
6. If a config is cheaper (shortened schedule), shorten it for **all** rows and say so.
7. Log the exact config diff per run; archive `args.yaml` alongside each CSV.
8. Report speed for the final model if the paper claims real-time suitability.
9. **Every run lives on its own `run/<name>` branch, committed and pushed before
   training starts (§6). No run from a dirty tree.**

---

## 9. Sign-off before writing the results section

- [ ] Every run traceable to a pushed `run/<name>` branch and commit SHA (§6)
- [ ] Base model decided and justified (§0)
- [ ] All §3 rows complete at matched settings
- [ ] Seed variance quantified; noise floor stated
- [ ] Test set evaluated **once**, on the final model only
- [ ] Split provenance confirmed (official vs custom) — determines whether the
      comparison to Nguyen et al. is a direct benchmark or indicative only
- [ ] Inference speed measured (if claiming real-time)
- [ ] Every borrowed component cited to its **original** source
      (CBAM → Woo 2018; BoT → Srinivas 2021; SIoU → Gevorgyan 2022;
      NWD → Wang 2021; WIoU-v3 → Tong 2023 — do **not** inherit Shi et al.'s
      citations, two of theirs point to application papers)