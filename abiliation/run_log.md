# Ablation run log

One row per run (§6 of study_specifications.md). Fill AP/F1 from the run's
test row after the final evaluation.

| run_name | branch | commit SHA | seed | epochs | best epoch | pollen AP@0.5:0.95 | pollen F1 | notes |
|---|---|---|---|---|---|---|---|---|
| abl_baseline | run/abl_baseline | bc57176 | 42 | 100 | 78 | 0.699 | 0.919 | stock YOLOv8m: attn=none, CIoU, stock BCE. Val (ranking, §8): AP@0.5:0.95 0.672, F1 0.896 |
