# Method description

The submitted system is five landmark models whose heatmaps are averaged before decoding.
This document describes what each model is, how they are combined, and which parts of the
system carry a fitted parameter. The code that implements everything below is `docker/model.py`
plus `docker/vendor/`. The corresponding paper is "Multi-Task Landmark Detection for Fetal and
Cardiac Ultrasound Biometry under Label Scarcity" (Medellin-Robles, Juarez-Castro, Bukhari),
published in the MICCAI 2026 Workshops and Challenges proceedings (FoundUS satellite event,
paper 92), open access at https://papers.miccai.org/miccai-2026-sat/FoundUS_092.html.

## 1. Shared substrate

Every member is an encoder with nine task heads. Each head emits one 64x64 heatmap per
landmark (A4C 16, AOP 4, FA 4, FUGC 2, HC 4, IVC 2, PLAX 22, PSAX 4, fetal femur 2), not
coordinates. Targets are Gaussian blobs with fixed sigma 1.8 cells and the loss is MSE on
the sigmoid of the output.

Training runs in two stages: 5 epochs with the encoder frozen, then 35 epochs with
everything trainable and the encoder learning rate 5 times lower than the head learning
rate. The weights we ship are an EMA copy with decay 0.999.

Unlabeled images enter in the second stage through weak-to-strong consistency, gated per
task at the 25th percentile of teacher confidence. The resulting cutoffs differ by close to an
order of magnitude across tasks: on the curation member they run from 0.058 on FA to 0.493 on
fetal femur, and on the anchor member from 0.089 on A4C to 0.531 on fetal femur. A per-task
gate is therefore not a refinement of a global one. On most tasks it is the difference between
gating and not gating at all.

The two-stage schedule above holds for four members. The curation member scales both learning
rates by the square root of its larger batch, which preserves the 5:1 ratio between head and
encoder but not the absolute values. The batch and the reference batch are set in
`training_code/DINOv2+Curation/run_v4_umv2.sh` and the scaling rule is in
`nb06_train_r2_engine.py`.

Heatmap output is what makes fusion possible: all five members emit the same grid in the same
channel order, so their outputs can be summed even though they differ in architecture, input
channels, resolution and normalization.

## 2. The five members

The paper uses reader-friendly names for the same five checkpoints. The mapping, so the two can
be read side by side:

| Name in this document | Name in the paper | Training code | Checkpoint |
|---|---|---|---|
| nb04 | DINOv2+Cons | `training_code/DINOv2+Cons/` | `nb04.pth` |
| m2_v4 | DINOv2+Curation | `training_code/DINOv2+Curation/` | `m2_v4.pth` |
| ultradino_ivc | DINOv2+SSL | `training_code/DINOv2+SSL/` | `ultradino_ivc.pth` |
| r3_a4c | DINOv2+Cons-B | `training_code/DINOv2+Cons-B/` | `r3_a4c.pth` |
| usfm | USFM | `training_code/USFM/` | `usfm.pth` |

The short internal names are the ones `docker/model.py` uses, so they are kept in this document.

The assembled system is called D7 internally and BloomMeasure in the paper.

Members were chosen so that their errors would be as uncorrelated as possible. Each is one
deliberate departure from the reference model.

**nb04** is the reference: DINOv2 ViT-S/14 at 518, trained on the full unlabeled corpus
without differentiation. Alone, with scale test-time augmentation, it scores 24.28 on public
validation.

**m2_v4** uses the same architecture and schedule but routes the unlabeled corpus instead of
pooling it. See section 3.

**ultradino_ivc** continues DINOv2 self-supervised pre-training (DINO, iBOT, KoLeo) for 40
epochs on the 187,744 unlabeled challenge images, with the patch embedding folded to a
single channel and no contrast equalization. Alone it is in the bottom third on all nine tasks
by radial error, and it is in the ensemble anyway: removing it costs 7.08 px on IVC and 3.13 px
on fetal femur on the internal split, measured on three separate builds.

**r3_a4c** is the reference with A4C excluded from the consistency loss. The A4C unlabeled
pool is two acquisition domains under one label (2,238 raw images at 42% Doppler, 61,152
pre-processed patches at 0%), and consistency forces agreement between populations that
should not agree. This member enters the ensemble for one task only, fetal femur.

**usfm** is the only non-DINOv2 member: BEiT-B/16 with public ultrasound pre-trained
weights, supervised only, 512x512 with a 32x32 patch grid, mean and std both 0.5. It is the
weakest member alone at 27.79, and that is not why it is in the ensemble. PSAX fails from a
lack of in-domain information rather than from the decoder or the training signal, and an
ultrasound pre-trained encoder moves that task from 42.16 to 34.58 px (hard-routed to USFM alone; the submitted 0.60 blend scores 34.53 on validation), and the same direction
held every time we rebuilt it.

## 3. Curated semi-supervision, member m2_v4

Standard semi-supervised training treats the 191,170 unlabeled images identically. They are
not identical. Some are easy enough that the model's own prediction is a more informative
target than a consistency constraint, some are genuinely ambiguous, and some are broken
(wrong subdomain, corrupted, filed under the wrong task) and hurt training under either
treatment. We measure every image before training and route it.

The self-supervised stage of DINOv2+SSL used 187,744 of these images, the rows that survived its own manifest build; the drop log for the 3,426 difference is in the image, not in this repository.

A trained model is frozen and used only as an instrument. Every image is passed six times under
photometric test-time augmentation (identity, gamma 0.85, gamma 1.15, brightness plus and minus
0.15, and a 3-pixel Gaussian blur; none of these move a pixel, so decoded coordinates need no
inverse mapping), which gives a mean prediction, an uncertainty (the spread of the six answers),
a peak confidence, and an embedding. Three shape signals are
added: a per-task geometric prior over distances, ratios and angles, fitted on the most
confident images by median and MAD so outliers cannot drag it; a neighbour score, the
distance from the shape agreed on by the image's 10 nearest embedding neighbours within its
task; and an embedding outlier score. All five signals become within-task percentiles, so no
threshold has to transfer across tasks of different difficulty.

| Route | Condition | Treatment |
|---|---|---|
| Discard | any of geometric prior, uncertainty, outlier score above the 97th percentile | removed from training |
| Pseudo-label | confidence above the 70th and uncertainty below the 50th and geometry below the 90th and neighbour below the 90th and outlier below the 95th | supervised on the model's own prediction, weight in [0.4, 1.0] by design; realized range 0.669 to 0.997 |
| Consistency | everything else | weak-to-strong consistency only |

The asymmetry is deliberate. One extreme signal is enough to discard, but promotion requires
all five to agree, because an image wrongly sent to consistency loses a little signal while
an image wrongly promoted injects a falsehood as ground truth and is then trained on with a
weight. Every image takes exactly one route, which is enforced in code.

IVC is capped at three times its labeled training rows, so 90 pseudo-labels from 255 eligible
candidates, and A4C is capped to a domain-matched slice, 2,172 of 15,381. Both caps exist so the
model's own predictions cannot bury a small set of human labels: an earlier build gave IVC 388
pseudo-labels against 30 labeled images, from a pool that was roughly 25% split-screen or Doppler,
and its leaderboard IVC score went from 25.01 to 39.09 px.

Every real label is kept at weight 1.0. This member scored 191,169 of the 191,170 unlabeled
images and routed them as follows: 27,122 added as pseudo-labels, 131,330 used for consistency
only, 32,717 discarded. Discarded here means excluded from this member's training, not excluded
from the system: of the other two members that carry a consistency branch, nb04 consumes the
unlabeled corpus whole and r3_a4c consumes it whole apart from its deliberate A4C exclusion, and
ultradino_ivc consumes all 187,744 of its self-supervised images regardless of this routing.

## 4. Fusion

For landmark k, member m emits a 64x64 map H_m with sigmoid values in (0,1). The fused map
is the weighted sum of the member maps with weights summing to one, and a single DARK decode
is applied to the fused map. One decode for the ensemble, not one decode per member.

Two properties follow. First, writing each map as the true field plus zero-mean noise, the
fused noise variance for M equal-weight members with per-member variance s and average
pairwise correlation rho is s(1 + (M-1)rho)/M. The reduction is bought by decorrelation, not
by member quality, which is why the members were chosen for architectural diversity rather
than by picking the five best checkpoints. Mean pairwise disagreement is 8.15 px between the two
members that differ by one switch and 16.1 to 17.5 px for every pair that includes the
non-DINOv2 member. The statistic the paper publishes for the same property is the
leave-one-member-out output shift: 5.21 px under uniform averaging against 7.71 px under
per-task routing, over the eight tasks where both have redundancy.

Second, summing dense maps and decoding once means a sharp, tall peak pulls the fused peak
further than a flat one. The confident member dominates without any confidence estimate
being computed, calibrated or tuned. Averaging decoded coordinates instead gives every
member an equal vote regardless of certainty, and a confidently wrong outlier then drags the
result to a wrong midpoint.

Weight-space merging was not rejected, it was unavailable: ultradino_ivc has a single-channel
patch embedding and usfm is a different architecture family, so there is no shared parameter
space for model soup, TIES or Fisher merging.

Each member is forwarded once at its own resolution, with no per-member test-time
augmentation.

## 5. Decoding

A 64x64 map read by argmax is quantized. DARK models the peak as a 2D Gaussian and takes one
Newton step on the log map, with the map lightly Gaussian-smoothed at sigma 1.8. The
correction is clipped to half a cell: if it wants to be larger, the brightest cell was the
wrong cell, which is a real error rather than a rounding one, so the argmax is kept. The same
fallback applies at the boundary.

The pure quantization floor is 0.0177 px against an MRE near 23, so DARK is not about
precision. What it removes is a bias, and that bias is not neutral between the alternatives
we compared: a fusion-rule comparison changed by more than a factor of two depending on
whether decoding was sub-pixel. Raising the grid to 128 or 256 makes results worse.

## 6. Anatomical projection on FA

FA has four landmarks that are the endpoints of two perpendicular ellipse axes sharing a
centre. Stacking the eight coordinates as a vector x, that structure is a set of linear
constraints Cx = 0 with a 4x8 matrix C encoding centre-sharing and symmetry. We apply the
orthogonal projector onto the null space of C, P = I - C'(CC')^{-1}C, verified symmetric and
idempotent to 1e-9.

The bound is formal rather than empirical. Non-expansiveness gives ||Px - Px_gt|| <= ||x -
x_gt||, and since the ground truth itself violates the constraint slightly, the bound against
the truth carries that violation as slack: the step cannot increase the error by more than the
ground-truth violation, which we measure at 0.250 px on the FA labels. In practice it reduced
the error on every build we measured. On the 188 FA validation images the post-projection
violation is 0.000000 px and points move a median of about 6 px.

The same step fails on HC and is not applied there. HC axes are rotated a median of 149
degrees from the image axes, so the true constraint is C(Rx) = 0 for an unknown rotation R,
and the axis-aligned projector projects onto the wrong subspace. HC error is rotational
(49%) rather than translational (5.2%), priced at 3.61 px per degree. An oracle given the
true angle buys 8 px, but every angle estimator we built buys a fraction of that or loses.
HC is left alone.

## 7. Per-task weights

| Tasks | nb04 | r3_a4c | ultradino_ivc | usfm | m2_v4 |
|---|---|---|---|---|---|
| A4C, AOP, FA, FUGC, HC, IVC, PLAX | 0.250 | 0 | 0.250 | 0.250 | 0.250 |
| PSAX | 0.133 | 0 | 0.133 | 0.600 | 0.133 |
| fetal femur | 0.333 | 0.333 | 0.333 | 0 | 0 |

Seven of nine tasks are a plain equal average of four members with no fitted quantity
anywhere. Two tasks carry a rule, and both had to be justified by a mechanism and had to stay
redundant.

PSAX raises the ultrasound pre-trained member to 0.60 rather than handing it the task
outright. The 0.40 left to the others is deliberate redundancy, so that if usfm degrades on
the private test three members absorb the failure instead of none. Diluting usfm to one fifth
of the ensemble still preserves 69% of its PSAX advantage, so the compromise is cheap. Fetal
femur goes to the three DINOv2 members because usfm and m2_v4 are independently weaker there;
it is still an average of three, not a single model.

One global constraint governs both rules: at most floor((M-1)/2) non-DINOv2 members may
contribute to any fused heatmap. In the build script that composed the candidates this was an
assertion that aborted on an illegal member set; in the submitted container the member sets are
fixed constants at `docker/model.py:63-71`.

These four numbers are the only quantities in the system fitted on the public leaderboard.
Everything else is either an equal average or a closed-form step.

## 8. Training data

All supervised training used challenge labeled data only, split into a training portion and a
held-out portion used for model selection. The split changed size once: on 2026-07-23 the
organizers withdrew 25 fetal_femur images as invalid, taking the labeled pool from 6,768 to
6,743 and the split from 5,407 / 1,361 to 5,386 / 1,357. Members trained before that date saw
the larger version, the curation member was trained after it and saw the smaller one, and no
image moved between the two portions. `training_code/splits/` ships the post-withdrawal
version. All semi-supervised training used the challenge
unlabeled corpus only. Two publicly released pre-trained encoders were used as
initialization: DINOv2 ViT-S/14 (LVD-142M) and USFM (BEiT-B/16, ultrasound). No external
image dataset was added to training, and no additional annotation of any public data was
produced. `docs/COMPLIANCE.md` states this rule by rule.

## 9. One implementation detail we chose to state

The single-channel member `ultradino_ivc` was pre-trained and fine-tuned under one set of
grayscale statistics (mean 0.1066, recorded in
`training_code/DINOv2+SSL/ours_norm_stats.json`), while inference runs it under a
slightly different set pinned in the shipped transform code (mean 0.1150,
`docker/vendor/src/training/albu_transforms.py`). The mechanism is a failed path lookup after a
directory move: the statistics file the code expected was no longer where it looked, and the code
fell back to a second statistics file measured on a different internal experiment folder. We
found it while auditing the container, not while building it.

Two things a reader diffing the repository will notice. The training-side copy of that transform file,
`training_code/DINOv2+SSL/albu_transforms_TRAINING.py`, resolves the statistics by path at import time while the container copy
pins them as constants, so a retrain from `training_code/` will not automatically reproduce the
inference-time values; use the pinned ones. And the fallback path and the constant are labelled
"EchoCare" and "folder 03", which are names of our own internal experiment directories, not of an
external dataset.

We did not change it. The submitted predictions, the public validation score of 22.5822 and every
ablation reported here were all produced under the pinned inference statistics, so that
configuration is the one that was measured. Changing it after the fact would have shipped a system
that no number in this repository describes.

Related imprecision in the paper, noted here rather than left to be found: section 3.2 describes
the DINOv2 members collectively as using CLAHE, percentile clipping and ImageNet normalization.
That holds for three of the four. This single-channel member uses corpus grayscale statistics and
no contrast equalization, as section 2 of this document states.

## 10. How the system was composed

Every step was checked on the public leaderboard before the next one was built, so the chain
itself is the evidence that each step transfers.

| Step | What changed | Public LB avg MRE |
|---|---|---:|
| Single model champion | DINOv2+Cons with scale test-time augmentation | 24.2808 |
| Three-model ensemble | heatmap average of DINOv2+Cons, DINOv2+Cons-B, DINOv2+SSL | 23.7149 |
| PSAX routed | PSAX taken by the ultrasound pre-trained encoder | 23.24 |
| Curation member folded in | DINOv2+Curation added to the tasks it improves | 23.0077 |
| FA ellipse projection | closed-form anatomical projection on FA | 22.9527 |
| PSAX blend and femur trio | the two per-task weight rules | 22.5822 |
| Organizer baseline, for reference | | 38.4426 |

Two of these steps are per-task rules rather than plain averaging. The reference point without
any of them is a uniform average of all five members, which scores 23.3436.

One caveat the paper carries on the same table and that belongs here too: the rows between the
three-model ensemble and the FA projection were built on an earlier revision of the curation
member, which the submitted system later replaced. They are the lineage of the method, not
runs of the submitted system.

The first step was the largest and the cheapest: averaging heatmaps instead of selecting one
model. It is image-agnostic variance reduction rather than a fit to the validation set, and
it was the first improvement in the project that survived contact with the leaderboard.

## 11. Ensemble size ablation

All 31 non-empty subsets of the five members, each averaged uniformly, decoded once, scored
on a task-stratified internal subset capped at 60 images per task, 354 images in total, then re-scored under 12
corruptions (acoustic shadowing, structured speckle, motion blur, gamma darkening, three
severities each). Degradation is the corrupted score minus that subset's own clean score, so
each subset is its own control.

| Members | Clean mean | Clean spread | Degradation mean | Degradation spread |
|---|---:|---:|---:|---:|
| 1 | 26.6350 | 2.73 | 7.1960 | 8.14 |
| 2 | 25.7354 | 2.16 | 5.0293 | 3.71 |
| 3 | 25.2646 | 1.36 | 4.5171 | 1.22 |
| 4 | 25.2452 | 0.97 | 4.1511 | 0.29 |
| 5 | 25.2103 | 0.00 | 3.8427 | 0.00 |

Both means improve monotonically with ensemble size and both spreads collapse. The second
column matters more than the first: at one model, 2.73 px of clean error and 8.14 px of
degradation ride on which model you happened to pick. A hand-picked trio beats the full
ensemble on clean data (24.64 against 25.21), but no subset of any size beats it under
corruption. Given a test set far larger and more varied than the 619 clean public images, we
took the degradation column.

## 12. What the fitted parameters cost

The submitted system carries four fitted weights, both of its per-task rules keep at least
three members contributing, and every other task is an equal average. That is the whole of the
tuning. What it risks is measurable: under the 12 corruptions, a uniform average of five
members degrades 3.843 px, the four DINOv2 members alone degrade 4.012 px, and hard per-task
routing degrades 4.567 px. Hard routing here means sending PSAX to a single member rather than
the 0.60 blend we submitted, so it is the pessimistic end of the range our rules sit in. The
loss under it is concentrated in PSAX, which degrades 4.13 px on average and 20.2 px under
severe acoustic shadowing, while a uniform average degrades essentially nothing there.

We kept the two conservative endpoints and report them rather than only our best number, with
the caveat from section 10 that both were measured on the earlier revision of the curation member.
A uniform average of all five members, with no per-task rule of any kind, scores 23.3436 on
public validation. Adding only the mechanism-driven steps, the in-domain PSAX routing and the
FA projection, gives 22.9527. The submitted system is 22.5822.

## 13. Limitations

- The corruptions in sections 11 and 12 are synthetic and are a proxy for acquisition degradation, not
  the real shift.
- Per-task counts in the internal analysis are small (PSAX 10 images, IVC 10), so individual
  magnitudes there are unreliable. What survives is direction and consistency.
- The seed noise floor is 0.307 px in the nine-task macro across three seeds of one identical
  configuration, and about 2 px on A4C alone. Any single training-side ablation smaller than
  that bar, including several of our own earlier ones, does not support a conclusion. This is
  why the ensemble argument rests on the monotone trend across 31 subsets rather than on any
  one pairwise comparison.

Final test results are in the repository `README.md`.
