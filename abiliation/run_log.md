# Ablation run log

One row per run (§6 of study_specifications.md). Fill AP/F1 from the run's
test row after the final evaluation.

| run_name | branch | commit SHA | seed | epochs | best epoch | pollen AP@0.5:0.95 | pollen F1 | notes |
|---|---|---|---|---|---|---|---|---|
| abl_baseline | run/abl_baseline | bc57176 | 42 | 100 | 78 | 0.699 | 0.919 | stock YOLOv8m: attn=none, CIoU, stock BCE. Val (ranking, §8): AP@0.5:0.95 0.672, F1 0.896 |
| abl_cbam | run/abl_cbam | e3856ff | 42 | 100 | 87 | 0.714 | 0.942 | row 1: CBAM alone (POLLENBEES_ATTN=cbam), CIoU, stock BCE — only delta vs baseline. Val (ranking, §8): AP@0.5:0.95 0.674, F1 0.905 — +0.002 AP vs baseline, within §7 noise floor. Test gains (AP +0.015, F1 +0.023) driven by FP 7→1 |
| abl_bot | run/abl_bot | 1e17353 | 42 | 100 | 71 | 0.694 | 0.927 | row 2: BoT alone (POLLENBEES_ATTN=botnet), CIoU, stock BCE — only delta vs baseline (P5 C2f → BoTNet MHSA, 22.6M params). Val (ranking, §8): AP@0.5:0.95 0.686 — best of rows 0–2, +0.014 vs baseline (at edge of §7 noise floor), F1 0.896 (flat). Trades precision for recall (FAR 0.061→0.088). Note: test AP 0.694 < CBAM's — val and test disagree, rank on val |
| abl_saf | run/abl_saf | 2503504 | 42 | 100 | 79 | 0.716 | 0.948 | row 3: SAF-SIoU alone (args.yaml box_loss_type: saf_siou), attn none, stock BCE — only delta vs baseline. Val (ranking, §8): AP@0.5:0.95 0.680 (+0.008 vs baseline, below §7 noise floor), F1 0.911. Clean profile — recall 0.857→0.886, precision held (FAR 0.061→0.062, MR 0.143→0.114). Best test so far (AP 0.716, FP 5/FN 8) |
