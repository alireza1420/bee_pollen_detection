# SAF-SIoU: Size-Adaptive Focused SIoU Regression Loss

This document derives the bounding-box regression loss used in this work,
exactly as implemented in `utils/util.py` (`saf_siou_reg` and `BoxLoss`).
It is written for a reader seeing the loss for the first time: each component
is motivated, defined, and then assembled into the final training objective.

---

## 1. Motivation

IoU-based regression losses (CIoU, SIoU, …) share a weakness on small
objects: when a ground-truth box is only a few pixels wide, a localization
error of one or two pixels can drop the IoU from high overlap to zero. The
loss surface becomes discontinuous and near-flat exactly where the pollen
load — our minority, small-object class — lives. Conversely, distance-based
metrics that handle tiny boxes well (e.g. the normalized Wasserstein
distance) discard the precise overlap geometry that IoU-family losses
exploit for medium and large boxes.

SAF-SIoU resolves this tension by **blending the two regimes with a gate
that depends on the ground-truth box size**, and then applying a **dynamic
focusing factor** that concentrates the gradient on boxes of moderate
difficulty. It is built from three published ingredients — SIoU
[Gevorgyan, 2022], the normalized Gaussian Wasserstein distance (NWD)
[Wang et al., 2021], and WIoU-v3 dynamic focusing [Tong et al., 2023] —
composed in a novel size-adaptive arrangement.

## 2. Notation

All boxes are axis-aligned rectangles in **input-image pixels** (the
implementation converts predictions from per-level grid units to pixels
before evaluating the loss, because two hyperparameters below are in pixel
units). For a predicted box $B_p$ and its assigned ground-truth box $B_g$:

| Symbol | Meaning |
|---|---|
| $(x_p, y_p),\ (x_g, y_g)$ | box centers |
| $w_p, h_p,\ w_g, h_g$ | box widths and heights |
| $\Delta x = x_p - x_g,\quad \Delta y = y_p - y_g$ | center offsets |
| $c_w, c_h$ | width and height of the smallest box enclosing $B_p \cup B_g$ |
| $\mathrm{IoU} = \dfrac{|B_p \cap B_g|}{|B_p \cup B_g|}$ | intersection over union |

## 3. Component 1 — SIoU cost (geometry-aware overlap loss)

SIoU augments the IoU loss with two penalties, an **angle-modulated
distance cost** and a **shape cost**, so that the gradient tells the
predictor not only *that* it is misaligned but *in which direction* and
*in what aspect*.

**Angle cost.** Let $\sigma = \sqrt{\Delta x^2 + \Delta y^2}$ be the
center distance and

$$
\Lambda \;=\; \sin\!\bigl(2\arcsin \tfrac{|\Delta y|}{\sigma}\bigr).
$$

$\Lambda$ measures how far the line connecting the two centers deviates
from the horizontal or vertical axis: $\Lambda = 0$ when the offset is
purely axis-aligned and $\Lambda = 1$ at $45^\circ$. It does not appear as
a standalone term; it modulates the distance cost below, encouraging the
optimizer to first snap the prediction onto one axis and then close the
remaining gap along it — a shorter path than drifting diagonally.

**Distance cost.** With $\gamma = 2 - \Lambda$ and the offsets normalized
by the enclosing box,

$$
\Delta \;=\; \Bigl(1 - e^{-\gamma\,(\Delta x / c_w)^2}\Bigr)
\;+\; \Bigl(1 - e^{-\gamma\,(\Delta y / c_h)^2}\Bigr).
$$

When the centers are nearly axis-aligned ($\Lambda \to 0$, $\gamma \to 2$)
the penalty on the remaining offset is at its sharpest.

**Shape cost.** With relative side-length mismatches

$$
\omega_w = \frac{|w_p - w_g|}{\max(w_p, w_g)},\qquad
\omega_h = \frac{|h_p - h_g|}{\max(h_p, h_g)},
$$

the shape cost is

$$
\Omega \;=\; \bigl(1 - e^{-\omega_w}\bigr)^{\theta}
\;+\; \bigl(1 - e^{-\omega_h}\bigr)^{\theta},
\qquad \theta = 4,
$$

where $\theta$ controls how strongly aspect-ratio mismatch is punished; a
larger $\theta$ tolerates small mismatches and concentrates the penalty on
gross ones.

**SIoU loss.** The three parts combine as

$$
\mathcal{L}_{\mathrm{SIoU}}
\;=\; 1 - \mathrm{IoU} \;+\; \frac{\Delta + \Omega}{2}.
$$

*Implementation note.* The argument of $\arcsin$ is clamped to
$[0,\,1 - 10^{-6}]$: at a purely vertical or horizontal offset,
$|\Delta y| / \sigma$ rounds to exactly $1$ in float32, where $\arcsin$
has an infinite derivative and the backward pass would produce NaN.

## 4. Component 2 — Normalized Wasserstein distance (small-box similarity)

Following Wang et al. (2021), each box is modeled as a 2-D Gaussian whose
mean is the box center and whose standard deviations are half the box
sides: $B \mapsto \mathcal{N}\!\bigl(\mu, \Sigma\bigr)$ with
$\mu = (x, y)$ and $\Sigma = \mathrm{diag}\bigl((w/2)^2, (h/2)^2\bigr)$.
This reflects the fact that an annotated rectangle is a soft container:
the object concentrates near the center, and the boundary pixels are
partly background.

The squared 2-Wasserstein distance between the two Gaussians has the
closed form

$$
W_2^2(B_p, B_g)
\;=\; \Delta x^2 + \Delta y^2
\;+\; \frac{(w_p - w_g)^2 + (h_p - h_g)^2}{4},
$$

which is then mapped to a bounded similarity in $(0, 1]$:

$$
\mathrm{NWD} \;=\; \exp\!\Bigl(-\frac{W_2}{C}\Bigr),
\qquad C = 12.8 \text{ px}.
$$

Unlike IoU, $W_2$ is a smooth metric on box parameters: it changes
proportionally to the pixel error **even when the boxes do not overlap at
all**, which is precisely the regime where tiny-object regression breaks
down under IoU-family losses. The constant $C$ sets the pixel scale at
which similarity decays and is a dataset-level hyperparameter.

## 5. Component 3 — the size gate (blending the two regimes)

The two costs are blended by a gate driven by the **ground-truth** box
area:

$$
\lambda \;=\; \exp\!\Bigl(-\frac{\sqrt{w_g\, h_g}}{T}\Bigr),
\qquad T = 32 \text{ px},
$$

$$
\boxed{\;
\mathcal{L}_{\mathrm{reg}}
\;=\; (1 - \lambda)\,\mathcal{L}_{\mathrm{SIoU}}
\;+\; \lambda\,\bigl(1 - \mathrm{NWD}\bigr)
\;}
$$

$\sqrt{w_g h_g}$ is the side length of the square with the same area as
the ground-truth box, so $\lambda$ is a smooth, monotone function of
object size:

| GT box | $\sqrt{w_g h_g}$ | $\lambda$ | dominant term |
|---|---|---|---|
| $16 \times 16$ | 16 px | $0.61$ | NWD |
| $32 \times 32$ | 32 px | $0.37$ | mixed |
| $64 \times 64$ | 64 px | $0.14$ | SIoU |
| $128 \times 128$ | 128 px | $0.02$ | SIoU |

Small boxes are therefore regressed chiefly under the smooth Wasserstein
similarity, large boxes under the geometry-aware SIoU cost, and the
transition is continuous — no hard size threshold, no separate branches.
The gate uses the ground-truth size (not the predicted size) so that it is
constant with respect to the parameters being optimized and cannot be
gamed by shrinking or inflating predictions.

## 6. Component 4 — WIoU-v3 dynamic non-monotonic focusing

Not every assigned box deserves the same gradient. Very easy boxes are
already solved; extremely hard ones are often unmatchable (occlusion,
annotation noise) and dragging the model toward them harms the average
case. Following WIoU-v3, each box receives a focusing factor computed
from its **outlier degree** — its loss relative to the running average
loss:

$$
\beta \;=\; \frac{\mathcal{L}_{\mathrm{reg}}^{\;*}}{\overline{\mathcal{L}}_{\mathrm{reg}}},
\qquad
r \;=\; \frac{\beta}{\delta\,\alpha_f^{\;\beta - \delta}},
\qquad \alpha_f = 1.9,\ \delta = 3,
$$

where $^*$ denotes a detached (no-gradient) copy, and
$\overline{\mathcal{L}}_{\mathrm{reg}}$ is an exponential running mean of
the batch-mean regression loss, updated only during training:

$$
\overline{\mathcal{L}}_{\mathrm{reg}} \;\leftarrow\;
m\,\overline{\mathcal{L}}_{\mathrm{reg}} + (1 - m)\,
\mathrm{mean}\bigl(\mathcal{L}_{\mathrm{reg}}^{\;*}\bigr),
\qquad m = 0.999 .
$$

The factor $r$ is **non-monotonic** in $\beta$: it is small for very easy
boxes ($\beta \to 0$), rises to its maximum ($\approx 1.3$ near
$\beta \approx 1.6$ with the values above), equals $1$ at $\beta = \delta$,
and decays again for outliers ($\beta \gg \delta$). Gradient is thus
concentrated on *ordinary-quality* boxes — the ones where effort pays —
while both trivial and pathological boxes are down-weighted.

Because $\overline{\mathcal{L}}_{\mathrm{reg}}$ shrinks as training
progresses, the definition of "ordinary" adapts over time: a box that
counts as an outlier early in training gradually re-enters the focused
band as the model improves. Two details matter for correctness:

- $r$ is treated as a constant during backpropagation (it is computed
  from detached values), so focusing reweights gradients without adding
  spurious gradient paths;
- $\overline{\mathcal{L}}_{\mathrm{reg}}$ is stored as a model buffer and
  saved in checkpoints, so a resumed run continues with a consistent
  notion of difficulty rather than resetting it.

## 7. Assembled objective

Let $\mathcal{F}$ be the set of foreground anchors selected by the
task-aligned assigner, and $w_i$ the assigner's alignment score for anchor
$i$ (the standard YOLOv8 per-anchor weight). The box term of the training
loss is

$$
\mathcal{L}_{\mathrm{box}}
\;=\; \frac{\sum_{i \in \mathcal{F}} r_i\,
\mathcal{L}_{\mathrm{reg},i}\, w_i}{\sum_i s_i},
$$

where $\sum_i s_i$ is the total assigned score (the same normalizer
YOLOv8 uses for all loss terms). The complete objective keeps YOLOv8's
structure and gains:

$$
\mathcal{L} \;=\;
\lambda_{\mathrm{box}}\,\mathcal{L}_{\mathrm{box}}
+ \lambda_{\mathrm{cls}}\,\mathcal{L}_{\mathrm{cls}}
+ \lambda_{\mathrm{dfl}}\,\mathcal{L}_{\mathrm{dfl}},
\qquad
(\lambda_{\mathrm{box}}, \lambda_{\mathrm{cls}}, \lambda_{\mathrm{dfl}})
= (7.5,\ 0.5,\ 1.5),
$$

with $\mathcal{L}_{\mathrm{cls}}$ a Varifocal loss carrying a fixed
minority-class positive weight (imbalance handling, described separately)
and $\mathcal{L}_{\mathrm{dfl}}$ the standard Distribution Focal Loss.
SAF-SIoU replaces only the pairwise box cost inside
$\mathcal{L}_{\mathrm{box}}$; the assigner, the DFL head, and the loss
normalization are unchanged from YOLOv8.

## 8. Hyperparameters

| Component | Symbol | Configuration key | Value | Role |
|---|---|---|---|---|
| Geometry term (SIoU) | $\theta$ | `saf_theta` | 4 | Shape-cost exponent |
| Small-box term (NWD) | $C$ | `saf_C` | 12.8 px | Similarity decay scale |
| Size gate | $T$ | `saf_T` | 32 px | Gate threshold |
| Dynamic focusing (WIoU-v3) | $\alpha_f$ | `saf_alpha_f` | 1.9 | Focusing base |
| Dynamic focusing (WIoU-v3) | $\delta$ | `saf_delta` | 3.0 | Focusing pivot ($r = 1$ at $\beta = \delta$) |
| Dynamic focusing (WIoU-v3) | $m$ | `saf_momentum` | 0.999 | Running-mean momentum |

$C$ and $T$ are expressed in input-scale pixels (the implementation
converts boxes to pixel units before evaluating the loss), so their
meaning is independent of which FPN level an anchor comes from.

## 9. Summary

$$
\mathcal{L}_{\mathrm{reg}}
= (1-\lambda)\underbrace{\Bigl[1 - \mathrm{IoU} + \tfrac{\Delta + \Omega}{2}\Bigr]}_{\text{SIoU: geometry for normal boxes}}
+ \lambda\underbrace{\Bigl[1 - e^{-W_2/C}\Bigr]}_{\text{NWD: smooth metric for tiny boxes}},
\qquad
\lambda = e^{-\sqrt{w_g h_g}/T},
$$

scaled per box by the WIoU-v3 factor $r(\beta)$ that focuses gradient on
boxes of ordinary difficulty. The size gate contributes robustness where
IoU fails (tiny pollen loads), SIoU contributes directional and shape
precision where overlap is informative, and the focusing factor allocates
capacity away from both solved and hopeless boxes.

## References

- Z. Gevorgyan, "SIoU Loss: More Powerful Learning for Bounding Box
  Regression," arXiv:2205.12740, 2022.
- J. Wang, C. Xu, W. Yang, L. Yu, "A Normalized Gaussian Wasserstein
  Distance for Tiny Object Detection," arXiv:2110.13389, 2021.
- Z. Tong, Y. Chen, Z. Xu, R. Yu, "Wise-IoU: Bounding Box Regression
  Loss with Dynamic Focusing Mechanism," arXiv:2301.10051, 2023.
