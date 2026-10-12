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

Deliberately unsupervised. There is no learned scorer here because fitting one
on this dataset means fitting and reporting on the same 1800 photographs, which
is exactly the leak documented as Leak 1. Unsupervised selection cannot leak,
because it never sees a label.
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

    def as_box(self):
        return (self.x, self.y, self.w, self.h)


# Feature order is fixed and asserted at load time, for the same reason
# classifier.py asserts DESCRIPTOR_LEN: silent feature drift would still produce
# predictions, just wrong ones.
SCORER_FEATURES = (
    "energy", "agree", "compact", "share", "contrast", "n_scales",
    "log_w", "log_h", "log_area", "log_aspect", "area_frac",
    "offx", "offy", "share_rank", "energy_rank",
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

    rows = []
    for i, c in enumerate(cands):
        area = max(1, c.w * c.h)
        rows.append([
            c.energy, c.agree, c.compact, c.share, c.contrast,
            c.n_scales / 9.0,
            np.log(max(1, c.w)), np.log(max(1, c.h)), np.log(area),
            np.log(max(1, c.w) / max(1, c.h)),
            area / total,
            (c.x + c.w / 2) / width - 0.5, (c.y + c.h / 2) / height - 0.5,
            share_rank[i], energy_rank[i],
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


def build_candidates(gray, cfg: ProposalConfig = None):
    """Propose at every scale and measure each candidate's three features.

    Returns Candidates whose `energy`, `agree` and `compact` are populated but
    whose `score` is not yet computed. Proposal generation is the expensive
    part and does not depend on the scorer weights, so a hyper-parameter search
    calls this once per image and then re-selects many times over it.
    """
    cfg = cfg or ProposalConfig()
    g = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    height, width = g.shape
    per_scale, energy_map, ref_mean = propose_all(g, cfg)
    ref_total = float(energy_map.sum()) + 1e-6
    ref_std = float(energy_map.std()) + 1e-6

    pool = [b for b in per_scale if len(b)]
    if not pool:
        return []
    all_boxes = np.concatenate(pool, axis=0)

    # Deduplicate identical boxes that several scales proposed.
    _, first_idx = np.unique(all_boxes, axis=0, return_index=True)
    uniq = all_boxes[np.sort(first_idx)]

    cands = []
    for bx, by, bw, bh in uniq:
        x, y, w, h = _clip(bx, by, bw, bh, width, height)
        if w < 2 or h < 2:
            continue
        box = (x, y, w, h)

        energy = float(energy_map[y:y + h, x:x + w].mean()) / ref_mean

        hits = 0
        for boxes in per_scale:
            if len(boxes) and _box_iou_vector(boxes, box).max() >= cfg.agree_iou:
                hits += 1
        agree = hits / max(1, len(per_scale))

        local = energy_map[y:y + h, x:x + w]
        active = local > local.mean()
        compact = float(active.mean())

        # Share of the frame's total anomaly captured by this box. A tight box on
        # a real defect takes a disproportionate slice; a box the size of a third
        # of the frame cannot score high however strong its mean is.
        share = float(local.sum()) / ref_total

        # Mean departure in standard deviations of the frame's anomaly spread.
        contrast = (float(local.mean()) - float(energy_map.mean())) / ref_std

        cands.append(Candidate(x, y, w, h, energy=energy, agree=agree,
                               compact=compact, share=share, contrast=contrast,
                               n_scales=hits))
    return cands


def select(cands, cfg: ProposalConfig = None, top_k=None, scorer=None, shape=None):
    """Apply the scorer, suppress overlaps, and return at most `top_k` regions.

    When a trained ProposalScorer is supplied its probability replaces the
    hand-weighted sum, because the learned score selects far better out of the
    multi-scale proposal set. Otherwise the unsupervised weights apply.
    """
    cfg = cfg or ProposalConfig()
    top_k = cfg.top_k if top_k is None else top_k
    if not cands:
        return []

    if scorer is not None:
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
    g = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    cands = build_candidates(g, cfg)
    return select(cands, cfg, top_k, scorer=scorer, shape=g.shape)