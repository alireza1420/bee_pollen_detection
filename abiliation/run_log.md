# Ablation run log

One row per run (§6 of study_specifications.md). Fill AP/F1 from the run's
test row after the final evaluation.

| run_name | branch | commit SHA | seed | epochs | best epoch | pollen AP@0.5:0.95 | pollen F1 | notes |
|---|---|---|---|---|---|---|---|---|
| abl_baseline | run/abl_baseline | bc57176 | 42 | 100 | 78 | 0.699 | 0.919 | stock YOLOv8m: attn=none, CIoU, stock BCE. Val (ranking, §8): AP@0.5:0.95 0.672, F1 0.896 |
| abl_cbam | run/abl_cbam | e3856ff | 42 | 100 | 87 | 0.714 | 0.942 | row 1: CBAM alone (POLLENBEES_ATTN=cbam), CIoU, stock BCE — only delta vs baseline. Val (ranking, §8): AP@0.5:0.95 0.674, F1 0.905 — +0.002 AP vs baseline, within §7 noise floor. Test gains (AP +0.015, F1 +0.023) driven by FP 7→1 |
| abl_bot | run/abl_bot | 1e17353 | 42 | 100 | 71 | 0.694 | 0.927 | row 2: BoT alone (POLLENBEES_ATTN=botnet), CIoU, stock BCE — only delta vs baseline (P5 C2f → BoTNet MHSA, 22.6M params). Val (ranking, §8): AP@0.5:0.95 0.686 — best of rows 0–2, +0.014 vs baseline (at edge of §7 noise floor), F1 0.896 (flat). Trades precision for recall (FAR 0.061→0.088). Note: test AP 0.694 < CBAM's — val and test disagree, rank on val |
| abl_saf | run/abl_saf | 2503504 | 42 | 100 | 79 | 0.716 | 0.948 | row 3: SAF-SIoU alone (args.yaml box_loss_type: saf_siou), attn none, stock BCE — only delta vs baseline. Val (ranking, §8): AP@0.5:0.95 0.680 (+0.008 vs baseline, below §7 noise floor), F1 0.911. Clean profile — recall 0.857→0.886, precision held (FAR 0.061→0.062, MR 0.143→0.114). Best test so far (AP 0.716, FP 5/FN 8) |
| abl_vflw | run/abl_vflw | 85baa12 | 42 | 100 | 75 | 0.707 | 0.937 | row 4: weighted VFL alone (args.yaml cls_loss_type: vfl + cls_pos_weight [1,2]), attn none, box back to CIoU — only delta vs baseline (§2: VFL+weighting as one bundle). Val (ranking, §8): AP@0.5:0.95 0.681 (+0.009 vs baseline, below §7 noise floor), F1 0.914. Recall-oriented as expected — recall 0.857→0.899, precision traded (FAR 0.061→0.071, MR 0.143→0.101). Test AP 0.707 / F1 0.937 (FP 7 / FN 9). Guard: NP-F1 0.983→0.981 |
| abl_no_cbam | run/abl_no_cbam | 06ce137 | 42 | 100 | | | | row 5 (leave-one-out, Full − CBAM): POLLENBEES_ATTN=botnet (BoT on, CBAM off) + SAF-SIoU + VFL-w. args.yaml identical to abl_full; only delta is dropping CBAM. Same backbone as abl_bot (~22.6M params), but with full-stack losses. Isolates CBAM's marginal gain on top of BoT+SAF+VFL — if AP ≈ abl_full within noise, CBAM is redundant. Awaiting run |
| abl_full | run/abl_full | 571e213 | 42 | 100 | | | | row 9 ★ FULL model — CBAM+BoT (attn=cbam_botnet) + SAF-SIoU + VFL-w. All 4 factors (env + 3 args.yaml). 22.71M params — LIGHTER than baseline (BoT replaces P5 C2f). Per §9 test is the reportable metric for the final model. Full stack de-risked locally. Awaiting run |
