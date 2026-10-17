"""Multi-scale defect localisation: propose at many scales, then select.

Replaces the single-threshold connected-component localiser in `neu.py`, whose
oracle recall at IoU 0.5 was measured at 0.082 - so no scorer, however good,
could have beaten macro F1 ~0.08. Diagnosis in docs/EVALUATION.md:

  * 88% of its boxes were more than 2x too large (median 7.3x in area);
  * NEU annotates 2.32 *separate* defect instances per image, median 11.8% of
    the frame each, so the annotations are tight and well separated;
  * the union of proposals over 21 (blur, threshold) combinations has oracle
    recall 0.476 @IoU0.5 - 5.8x the single-scale ceiling.

So the tight boxes were always in the proposal space and the single-threshold
design discarded them. This module keeps them by proposing at every scale and
then selecting with a score that needs no labels.

The score has three interpretable terms, all measured on real captures:

  energy   how far the pixels inside the box depart from their local background,
           normalised by the image mean so bright and dark frames compare.
  agree    how many of the scales independently proposed an overlapping box.
           A real defect survives several scales; texture noise usually
           appears at one and vanishes at the next.
  compact  component area / box area. A tight, solid region scores high; a
           box drawn around scattered specks scores low.

Selection is greedy non-maximum suppression by IoU, capped at `top_k`. NEU
averages 2.32 defects per image, so emitting more than a handful of boxes is
what destroyed precision in the previous version.

**Selection is now learned, not hand-weighted.** The unsupervised score above
could not exploit the proposal space: against an oracle of 0.3455 it reached only
0.21, because choosing 1 good box out of ~78 needs supervision. `ProposalScorer`
is a binary logistic regression trained on the TUNE split only and reported once
on a disjoint REPORT split, so it cannot leak the way a scorer fitted and scored
on the same images would (Leak 1). The hand-weighted terms above are retained as
the fallback path used when no scorer is supplied.

**And box size is now part of the search.** A second diagnosis
(`eval/diagnose_proposals.py`) found that 62.3% of missed ground-truth boxes were
the right defect at the wrong *size*, and only 2.6% were unseen - so the ceiling
was geometry, not detection. `add_variants` therefore grows and shrinks every
component about its anomaly centroid. See that function and docs/EVALUATION.md
section 4d for the measured effect (oracle 0.346 -> 0.546) and, equally
important, for why choosing the size with a label-free rule fails.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Reference scale for the `energy` term. Fixed rather than tuned so the score
# stays comparable across proposals drawn from other scales.
ENERGY_BOX = 21

SCORER_PATH = Path(__file__).resolve().parent / "scorer.npz"


@dataclass
class ProposalConfig:
    """Everything a multi-scale proposal run needs. All fields are tuned by
    eval/localize_tune.py on a TUNE split and reported on a disjoint split."""

    blur_sizes: tuple = (21, 31, 51)
    thresholds: tuple = (8.0, 18.0, 36.0)
    close_k: int = 5
    min_area: int = 20
    max_frac: float = 0.95

    # Nested box variants. Empty by default so the pre-variant behaviour - and
    # every number measured before this change - stays exactly reproducible.
    # See `add_variants` for why growing/shrinking is the right fix.
    variant_factors: tuple = ()

    # Selection
    nms_iou: float = 0.40
    top_k: int = 4

    # Scorer weights, on comparable 0..1 scales.
    w_energy: float = 1.0
    w_agree: float = 1.0
    w_compact: float = 0.5
    w_share: float = 0.0
    w_contrast: float = 0.0
    agree_iou: float = 0.50

    # Abstention: drop boxes scoring below this. Lets the localiser say "nothing
    # here" instead of always emitting its best guesses, which is what destroyed
    # precision in the single-scale version.
    score_thresh: float = 0.0


@dataclass
class Candidate:
    x: int
    y: int
    w: int
    h: int
    energy: float = 0.0
    agree: float = 0.0
    compact: float = 0.0
    share: float = 0.0
    contrast: float = 0.0
    score: float = 0.0
    n_scales: int = 0

    # Variant geometry. `var_factor` is 1.0 for a raw connected-component box.
    # `outside_mass` is the share of nearby anomaly energy sitting just OUTSIDE
    # the box, which is the signal for "this box is a fragment, grow it".
    var_factor: float = 1.0
    outside_mass: float = 0.0

    def as_box(self):
        return (self.x, self.y, self.w, self.h)


# Feature order is fixed and asserted at load time, for the same reason
# classifier.py asserts DESCRIPTOR_LEN: silent feature drift would still produce
# predictions, just wrong ones.
SCORER_FEATURES = (
    "energy", "agree", "compact", "share", "contrast", "n_scales",
    "log_w", "log_h", "log_area", "log_aspect", "area_frac",
    "offx", "offy", "share_rank", "energy_rank",
    # Size-decision features, added after eval/diagnose_proposals.py showed 62.3%
    # of ground-truth boxes were "refinement" misses - a proposal overlapping the
    # defect at the wrong size - and only 2.6% were unseen. A linear model cannot
    # represent "grow this box *because* there is anomaly just outside it", so the
    # interaction is supplied explicitly as `grow_interaction`.
    "var_log", "outside_mass", "grow_interaction",
)
SCORER_LEN = len(SCORER_FEATURES)


def candidate_matrix(cands, shape):
    """(N, SCORER_LEN) feature matrix for one image's candidates."""
    if not cands:
        return np.zeros((0, SCORER_LEN))
    height, width = shape
    total = max(1, height * width)
    shares = np.array([c.share for c in cands])
    energies = np.array([c.energy for c in cands])
    share_rank = np.argsort(np.argsort(-shares)) / max(1, len(cands) - 1)
    energy_rank = np.argsort(np.argsort(-energies)) / max(1, len(cands) - 1)
    # Normalise by the scales actually used in THIS image rather than a literal,
    # so changing ProposalConfig.blur_sizes or thresholds cannot silently
    # rescale this feature.
    n_scale_norm = max(1.0, float(max(c.n_scales for c in cands)))

    rows = []
    for i, c in enumerate(cands):
        area = max(1, c.w * c.h)
        var_log = float(np.log(max(1e-6, c.var_factor)))
        rows.append([
            c.energy, c.agree, c.compact, c.share, c.contrast,
            c.n_scales / n_scale_norm,
            np.log(max(1, c.w)), np.log(max(1, c.h)), np.log(area),
            np.log(max(1, c.w) / max(1, c.h)),
            area / total,
            (c.x + c.w / 2) / width - 0.5, (c.y + c.h / 2) / height - 0.5,
            share_rank[i], energy_rank[i],
            var_log, c.outside_mass, var_log * c.outside_mass,
        ])
    return np.asarray(rows, dtype=np.float64)


class ProposalScorer:
    """Binary logistic regression: is this proposal a real defect box?

    The unsupervised score above cannot exploit the multi-scale proposal space:
    measured against a 0.35 oracle recall ceiling it reached only 0.21, because
    "which of 78 boxes is right" needs supervision. This scorer learns that from
    the TUNE split only and is evaluated on a disjoint REPORT split, so it
    cannot leak the way a scorer fitted and scored on the same images would.

    Same hand-rolled NumPy gradient descent as classifier.py - this OpenCV 5
    wheel has no cv2.ml and the environment has no scikit-learn or torch.
    """

    def __init__(self, W, b, mu, sd, meta=None):
        self.W = W
        self.b = b
        self.mu = mu
        self.sd = sd
        self.meta = meta or {}

    def score(self, cands, shape):
        if not cands:
            return np.zeros(0)
        X = candidate_matrix(cands, shape)
        z = ((X - self.mu) / self.sd) @ self.W + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def save(self, path=SCORER_PATH):
        np.savez_compressed(path, W=self.W, b=self.b, mu=self.mu, sd=self.sd)

    @classmethod
    def load(cls, path=SCORER_PATH):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"no trained scorer at {path}. Run: python eval/train_scorer.py")
        d = np.load(path)
        if d["W"].shape[0] != SCORER_LEN:
            raise ValueError(
                f"scorer expects {d['W'].shape[0]} features but localize.py now "
                f"produces {SCORER_LEN}; retrain with eval/train_scorer.py")
        return cls(d["W"], d["b"], d["mu"], d["sd"])


def fit_scorer(X, y, epochs=600, lr=1.0, l2=1e-3, pos_weight=1.0):
    """Full-batch gradient descent on binary cross-entropy.

    `pos_weight` up-weights real boxes, because they are the minority class by a
    wide margin: most proposals in any image are texture, not defect.
    """
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-6
    Xn = (X - mu) / sd
    sw = np.where(y > 0.5, pos_weight, 1.0)
    sw = sw * (len(y) / max(1e-9, sw.sum()))
    W = np.zeros(Xn.shape[1])
    b = 0.0
    for _ in range(epochs):
        z = np.clip(Xn @ W + b, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        g = (sw * (p - y)) / sw.sum()
        W -= lr * (Xn.T @ g + l2 * W)
        b -= lr * g.sum()
    return W, b, mu, sd


def _components(mask, min_area, max_frac, total):
    """Connected components as an (N,4) int array of x,y,w,h."""
    n, _, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros((0, 4), dtype=np.int32), np.zeros((0,), dtype=np.float32)
    st = st[1:n]
    keep = (st[:, 4] >= min_area) & ((st[:, 2] * st[:, 3]) / total <= max_frac)
    st = st[keep]
    return st[:, :4].astype(np.int32), st[:, 4].astype(np.float32)


def _box_iou_vector(boxes, query):
    """IoU of every row of `boxes` against one (x,y,w,h) query."""
    x0 = np.maximum(boxes[:, 0], query[0])
    y0 = np.maximum(boxes[:, 1], query[1])
    x1 = np.minimum(boxes[:, 0] + boxes[:, 2], query[0] + query[2])
    y1 = np.minimum(boxes[:, 1] + boxes[:, 3], query[1] + query[3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    if not inter.any():
        return np.zeros(len(boxes), dtype=np.float64)
    area_b = (boxes[:, 2] * boxes[:, 3]).astype(np.float64)
    union = area_b + float(query[2] * query[3]) - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def _pairwise_iou(boxes):
    """Full IoU matrix for greedy NMS without an O(n^2) Python loop."""
    x0 = boxes[:, 0]
    y0 = boxes[:, 1]
    x1 = x0 + boxes[:, 2]
    y1 = y0 + boxes[:, 3]
    ix0 = np.maximum(x0[:, None], x0[None, :])
    iy0 = np.maximum(y0[:, None], y0[None, :])
    ix1 = np.minimum(x1[:, None], x1[None, :])
    iy1 = np.minimum(y1[:, None], y1[None, :])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    area = (boxes[:, 2] * boxes[:, 3]).astype(np.float64)
    union = area[:, None] + area[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def _clip(x, y, w, h, width, height):
    """Clip to the frame; blur at the border makes components run off the edge."""
    x0 = max(0, min(int(x), width - 1))
    y0 = max(0, min(int(y), height - 1))
    x1 = max(x0 + 1, min(int(x + w), width))
    y1 = max(y0 + 1, min(int(y + h), height))
    return x0, y0, x1 - x0, y1 - y0


def propose_all(gray, cfg: ProposalConfig = None):
    """Raw per-scale component boxes, plus the reference energy map.

    Returns (per_scale, energy_map) where per_scale is a list of (N,4) arrays,
    one per (blur, threshold) combination, in the order of cfg.
    """
    cfg = cfg or ProposalConfig()
    g = gray.astype(np.float32)
    total = g.shape[0] * g.shape[1]
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.close_k, cfg.close_k))

    ref_bg = cv2.blur(g, (ENERGY_BOX, ENERGY_BOX))
    energy_map = cv2.absdiff(g, ref_bg)
    ref_mean = float(energy_map.mean()) + 1e-6

    per_scale = []
    for k in cfg.blur_sizes:
        anom = cv2.absdiff(g, cv2.blur(g, (k, k)))
        for t in cfg.thresholds:
            # Thresholding a float map leaves it float; connectedComponentsWithStats
            # asserts CV_8U, so rescale to a real mask first.
            _, binary = cv2.threshold(anom, t, 1.0, cv2.THRESH_BINARY)
            mask = (binary * 255).astype(np.uint8)
            if cfg.close_k:
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
            boxes, _ = _components(mask, cfg.min_area, cfg.max_frac, total)
            per_scale.append(boxes)
    return per_scale, energy_map, ref_mean


def _integral_maps(energy_map):
    """Integral images of energy, energy*x and energy*y.

    Three integral images turn every per-box region statistic this module needs
    into four array lookups instead of a Python-level slice-and-sum. With
    variant boxes the candidate count tripled, so the old per-candidate loop was
    the slowest part of proposal building.
    """
    e = energy_map.astype(np.float64)
    h, w = e.shape
    xs = np.tile(np.arange(w, dtype=np.float64), (h, 1))
    ys = np.tile(np.arange(h, dtype=np.float64)[:, None], (1, w))
    return cv2.integral(e), cv2.integral(e * xs), cv2.integral(e * ys)


def _box_sums(ii, boxes):
    """Sum under each box of an (N,4) array, via an integral image."""
    if not len(boxes):
        return np.zeros(0, dtype=np.float64)
    x0 = boxes[:, 0].astype(np.int64)
    y0 = boxes[:, 1].astype(np.int64)
    x1 = x0 + boxes[:, 2].astype(np.int64)
    y1 = y0 + boxes[:, 3].astype(np.int64)
    return ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]


def _iou_matrix(a, b):
    """(N,4) by (M,4) IoU matrix."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float64)
    ax1 = a[:, 0].astype(np.float64) + a[:, 2]
    ay1 = a[:, 1].astype(np.float64) + a[:, 3]
    bx1 = b[:, 0].astype(np.float64) + b[:, 2]
    by1 = b[:, 1].astype(np.float64) + b[:, 3]
    ix0 = np.maximum(a[:, 0:1], b[None, :, 0])
    iy0 = np.maximum(a[:, 1:2], b[None, :, 1])
    ix1 = np.minimum(ax1[:, None], bx1[None, :])
    iy1 = np.minimum(ay1[:, None], by1[None, :])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    area_a = (a[:, 2] * a[:, 3]).astype(np.float64)[:, None]
    area_b = (b[:, 2] * b[:, 3]).astype(np.float64)[None, :]
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def add_variants(boxes, factors, energy_map, shape, integral=None):
    """Nested rectangles grown/shrunk about each box's anomaly centroid.

    Why this exists: `eval/diagnose_proposals.py` bucketed every missed
    ground-truth box and found 62.3% were "refinement" misses - a proposal
    already overlapping the defect, just at the wrong size - against only 2.6%
    that no proposal overlapped at all. So the ceiling was a *geometry* problem,
    not a detection problem, and more detector families would not have moved it.

    `eval/probe_factors.py` then chose these factors by greedy forward selection
    on the TUNE half, maximising oracle recall per extra box: oracle recall went
    0.351 -> 0.489 for 80 -> 240 boxes per image. Grow factors beat shrink
    factors because the common refinement miss is a component that is a
    *fragment* of the annotated defect, not a merge of several defects.

    Returns ((N,4) boxes, (N,) factors). Empty when `factors` is empty, which is
    what keeps the pre-variant behaviour reproducible.
    """
    if not len(boxes) or not factors:
        return np.zeros((0, 4), dtype=np.int32), np.zeros(0, dtype=np.float64)
    height, width = shape
    integral = integral or _integral_maps(energy_map)
    iiE, iiX, iiY = integral

    tot = _box_sums(iiE, boxes)
    # Anomaly-weighted centroid, falling back to the box centre when a box
    # contains no energy at all.
    cx = _box_sums(iiX, boxes) / np.maximum(tot, 1e-6)
    cy = _box_sums(iiY, boxes) / np.maximum(tot, 1e-6)
    empty = tot <= 1e-6
    cx = np.where(empty, boxes[:, 0] + boxes[:, 2] / 2.0, cx)
    cy = np.where(empty, boxes[:, 1] + boxes[:, 3] / 2.0, cy)

    out_b, out_f = [], []
    for f in factors:
        nw = np.maximum(2, np.rint(boxes[:, 2] * f)).astype(np.int64)
        nh = np.maximum(2, np.rint(boxes[:, 3] * f)).astype(np.int64)
        x0 = np.rint(cx - nw / 2.0).astype(np.int64)
        y0 = np.rint(cy - nh / 2.0).astype(np.int64)
        x0 = np.clip(x0, 0, np.maximum(0, width - 1))
        y0 = np.clip(y0, 0, np.maximum(0, height - 1))
        w_ = np.clip(x0 + nw, 1, width) - x0
        h_ = np.clip(y0 + nh, 1, height) - y0
        out_b.append(np.stack([x0, y0, np.maximum(1, w_), np.maximum(1, h_)], axis=1))
        out_f.append(np.full(len(boxes), float(f)))
    return (np.concatenate(out_b, axis=0).astype(np.int32),
            np.concatenate(out_f, axis=0))


def build_candidates(gray, cfg: ProposalConfig = None, proposals=None):
    """Propose at every scale, add nested variants, and measure every feature.

    Returns Candidates whose features are populated but whose `score` is not yet
    computed. Proposal generation does not depend on the scorer weights, so a
    hyper-parameter search calls this once per image and re-selects many times
    over it.

    `proposals` optionally supplies an already-computed `propose_all` result, so
    a caller that needs the anomaly map as well does not pay to build it twice.
    """
    cfg = cfg or ProposalConfig()
    g = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    height, width = g.shape
    per_scale, energy_map, ref_mean = (
        proposals if proposals is not None else propose_all(g, cfg))
    ref_total = float(energy_map.sum()) + 1e-6
    ref_std = float(energy_map.std()) + 1e-6
    emean = float(energy_map.mean())

    pool = [b for b in per_scale if len(b)]
    if not pool:
        return []

    # Deduplicate identical boxes that several scales proposed.
    all_boxes = np.concatenate(pool, axis=0)
    _, first_idx = np.unique(all_boxes, axis=0, return_index=True)
    base = all_boxes[np.sort(first_idx)]

    integral = _integral_maps(energy_map)
    iiE = integral[0]
    var_boxes, var_factors = add_variants(
        base, cfg.variant_factors, energy_map, g.shape, integral)
    if len(var_boxes):
        boxes = np.concatenate([base, var_boxes], axis=0)
        factors = np.concatenate([np.ones(len(base)), var_factors])
        # Dedupe on box AND factor: the same rectangle at two scales is two
        # distinct candidates, because `var_factor` is a scorer feature.
        keys = np.concatenate(
            [boxes, np.rint(factors[:, None] * 1000).astype(np.int64)], axis=1)
        _, keep = np.unique(keys, axis=0, return_index=True)
        keep = np.sort(keep)
        boxes, factors = boxes[keep], factors[keep]
    else:
        boxes, factors = base, np.ones(len(base))

    clipped = np.array([_clip(x, y, w, h, width, height) for x, y, w, h in boxes],
                       dtype=np.int32)
    keep = (clipped[:, 2] >= 2) & (clipped[:, 3] >= 2)
    boxes, factors = clipped[keep], factors[keep]
    if not len(boxes):
        return []

    n = len(boxes)
    areas = np.maximum(1.0, boxes[:, 2] * boxes[:, 3]).astype(np.float64)
    sums = _box_sums(iiE, boxes)
    means = sums / areas

    energy = means / ref_mean
    # Share of the frame's total anomaly captured by this box. A tight box on a
    # real defect takes a disproportionate slice; a box the size of a third of
    # the frame cannot score high however strong its mean is.
    share = sums / ref_total
    contrast = (means - emean) / ref_std

    # Anomaly energy in a ring just outside the box, relative to the energy
    # inside it. High means "the defect continues past this edge" - the signal
    # that a grown variant is the right rectangle. This is the feature that
    # lets the scorer turn the extra oracle recall into real recall.
    margin = np.maximum(3.0, 0.25 * np.maximum(boxes[:, 2], boxes[:, 3]))
    gx0 = np.clip(boxes[:, 0] - margin, 0, width).astype(np.int64)
    gy0 = np.clip(boxes[:, 1] - margin, 0, height).astype(np.int64)
    gx1 = np.clip(boxes[:, 0] + boxes[:, 2] + margin, 0, width).astype(np.int64)
    gy1 = np.clip(boxes[:, 1] + boxes[:, 3] + margin, 0, height).astype(np.int64)
    grown = np.stack([gx0, gy0, np.maximum(1, gx1 - gx0), np.maximum(1, gy1 - gy0)],
                     axis=1)
    ring = np.maximum(0.0, _box_sums(iiE, grown) - sums)
    outside_mass = ring / (sums + 1e-6)

    # How many DISTINCT scales independently proposed an overlapping box.
    allb = np.concatenate(pool, axis=0)
    scale_id = np.concatenate(
        [np.full(len(b), s, dtype=np.int32) for s, b in enumerate(per_scale)])
    hit = _iou_matrix(boxes, allb) >= cfg.agree_iou
    hits = np.zeros((n, len(per_scale)), dtype=bool)
    for s in range(len(per_scale)):
        hits[:, s] = hit[:, scale_id == s].any(axis=1)
    n_hit = hits.sum(axis=1)
    agree = n_hit / max(1, len(per_scale))

    cands = []
    for i in range(n):
        x, y, w, h = (int(v) for v in boxes[i])
        # compact: how much of the box is actually above its own local mean. A
        # tight solid region scores high; a box around scattered specks does not.
        local = energy_map[y:y + h, x:x + w]
        compact = float((local > local.mean()).mean()) if local.size else 0.0
        cands.append(Candidate(
            x, y, w, h,
            energy=float(energy[i]), agree=float(agree[i]), compact=compact,
            share=float(share[i]), contrast=float(contrast[i]),
            n_scales=int(n_hit[i]),
            var_factor=float(factors[i]), outside_mass=float(outside_mass[i])))
    return cands


def select(cands, cfg: ProposalConfig = None, top_k=None, scorer=None, shape=None,
           scores=None):
    """Apply the scorer, suppress overlaps, and return at most `top_k` regions.

    When a trained ProposalScorer is supplied its probability replaces the
    hand-weighted sum, because the learned score selects far better out of the
    multi-scale proposal set. Otherwise the unsupervised weights apply.

    `scores` lets a caller supply probabilities it has already computed. The
    probability does not depend on `top_k` or `nms_iou`, so a sweep over those
    two only has to pay for the feature matrix once per image instead of once per
    configuration - which is the difference between a minutes-long sweep and an
    hours-long one.
    """
    cfg = cfg or ProposalConfig()
    top_k = cfg.top_k if top_k is None else top_k
    if not cands:
        return []

    if scores is not None:
        for c, p in zip(cands, scores):
            c.score = float(p)
    elif scorer is not None:
        probs = scorer.score(cands, shape or (200, 200))
        for c, p in zip(cands, probs):
            c.score = float(p)
    else:
        # Each continuous feature is divided by its own max over this image's
        # candidates, so the weights mean the same thing on a dark frame as on a
        # bright one and no single feature's natural scale dominates.
        def norm(attr):
            top = max(getattr(c, attr) for c in cands)
            return [max(0.0, getattr(c, attr)) / (top + 1e-9) for c in cands]

        energy = norm("energy")
        share = norm("share")
        contrast = norm("contrast")
        for i, c in enumerate(cands):
            c.score = (cfg.w_energy * energy[i]
                       + cfg.w_agree * c.agree
                       + cfg.w_compact * c.compact
                       + cfg.w_share * share[i]
                       + cfg.w_contrast * contrast[i])

    order = np.argsort([-c.score for c in cands])
    order = [int(i) for i in order if cands[int(i)].score >= cfg.score_thresh]
    if not order:
        return []

    boxes = np.array([c.as_box() for c in cands], dtype=np.int32)
    iou_mat = _pairwise_iou(boxes)
    picked, suppressed = [], np.zeros(len(cands), dtype=bool)
    for idx in order:
        if suppressed[idx]:
            continue
        picked.append(cands[idx])
        suppressed |= iou_mat[idx] >= cfg.nms_iou
        if len(picked) >= top_k:
            break
    return picked


def localize(gray, cfg: ProposalConfig = None, top_k=None, scorer=None):
    """Propose at every scale, score, suppress, and return ranked regions.

    Greedy NMS in descending score order, so a returned list is ranked by
    confidence and the agent can escalate on the strongest region.
    """
    picked, _ = localize_ex(gray, cfg, top_k, scorer)
    return picked


def localize_ex(gray, cfg: ProposalConfig = None, top_k=None, scorer=None):
    """`localize` plus the anomaly energy map.

    The request path needs both the regions and a scalar summary of the frame's
    anomaly energy, and the energy map is already computed inside proposal
    generation - returning it avoids recomputing a blur and an absdiff.
    """
    g = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    cfg = cfg or ProposalConfig()
    proposals = propose_all(g, cfg)
    cands = build_candidates(g, cfg, proposals=proposals)
    return select(cands, cfg, top_k, scorer=scorer, shape=g.shape), proposals[1]