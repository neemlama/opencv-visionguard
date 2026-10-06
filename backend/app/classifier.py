"""NumPy multinomial logistic regression for NEU surface-defect classification.

Why hand-rolled: this OpenCV 5 wheel ships no cv2.ml module, and the environment
has neither scikit-learn nor torch. The features are OpenCV's; the decision
boundary is 3 lines of gradient descent. Adding a deep-learning framework for
six classes and 1800 images would be the wrong trade.

Trained offline by eval/train_classifier.py and stored as a .npz so inference
needs no training code at request time.

Reference held-out performance (stratified 70/30, seed 0, NEU-CLS 1800 images):
93.70% accuracy, per-class F1 0.86-0.99. See docs/EVALUATION.md.
"""
import json
from pathlib import Path

import numpy as np

from .features import CLASS_NAMES, DESCRIPTOR_LEN

MODEL_PATH = Path(__file__).resolve().parent / "model.npz"
META_PATH = Path(__file__).resolve().parent / "model_meta.json"


class DefectClassifier:
    def __init__(self, W, b, mu, sd, class_names=CLASS_NAMES, meta=None, temperature=1.0):
        self.W = W
        self.b = b
        self.mu = mu
        self.sd = sd
        self.class_names = tuple(class_names)
        self.meta = meta or {}
        # Temperature fitted on the held-out split. Raw softmax output was
        # systematically UNDER-confident (stated 0.79 where accuracy was 0.94,
        # ECE 0.075), which made the agent escalate far too little. Dividing the
        # logits by T sharpens when T<1; the fitted value here is 0.70.
        # See fit_temperature().
        self.temperature = float(temperature)

    # ---------- inference ----------
    def predict_proba(self, feats):
        x = (np.atleast_2d(feats) - self.mu) / self.sd
        z = (x @ self.W + self.b) / max(self.temperature, 1e-6)
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict_one(self, feats):
        p = self.predict_proba(feats)[0]
        idx = int(p.argmax())
        # Sorted margin over the runner-up: the signal the agent thresholds on.
        srt = np.sort(p)[::-1]
        return {
            "defect_class": self.class_names[idx],
            "confidence": float(p[idx]),
            "margin": float(srt[0] - srt[1]) if len(srt) > 1 else float(srt[0]),
            "probs": {c: round(float(v), 4) for c, v in zip(self.class_names, p)},
        }

    # ---------- persistence ----------
    def save(self, path=MODEL_PATH, meta_path=META_PATH):
        np.savez_compressed(path, W=self.W, b=self.b, mu=self.mu, sd=self.sd,
                            temperature=np.array([self.temperature]))
        Path(meta_path).write_text(json.dumps(
            {"class_names": list(self.class_names),
             "temperature": self.temperature,
             **(self.meta or {})}, indent=2))

    @classmethod
    def load(cls, path=MODEL_PATH, meta_path=META_PATH):
        path, meta_path = Path(path), Path(meta_path)
        if not path.exists():
            raise FileNotFoundError(
                f"no trained model at {path}. Run: python eval/train_classifier.py")
        d = np.load(path)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        names = tuple(meta.pop("class_names", CLASS_NAMES))
        temp = float(meta.pop("temperature", 1.0))
        if "temperature" in d:
            temp = float(d["temperature"][0])
        # Guard against silent feature drift: a mismatched descriptor would still
        # produce predictions, just wrong ones.
        if d["W"].shape[0] != DESCRIPTOR_LEN:
            raise ValueError(
                f"model expects {d['W'].shape[0]} features but features.py now "
                f"produces {DESCRIPTOR_LEN}; retrain with eval/train_classifier.py")
        return cls(d["W"], d["b"], d["mu"], d["sd"], names, meta, temp)


def fit(X, y, n_classes, epochs=400, lr=0.5, l2=1e-3):
    """Full-batch gradient descent on multinomial cross-entropy."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-6
    Xn = (X - mu) / sd
    n, d = Xn.shape
    W = np.zeros((d, n_classes))
    b = np.zeros(n_classes)
    Y = np.zeros((n, n_classes))
    Y[np.arange(n), y] = 1.0
    for _ in range(epochs):
        Z = Xn @ W + b
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)
        G = (P - Y) / n
        W -= lr * (Xn.T @ G + l2 * W)
        b -= lr * G.sum(axis=0)
    return W, b, mu, sd


def loss(Xn, y, W, b):
    Z = Xn @ W + b
    Z -= Z.max(axis=1, keepdims=True)
    lse = np.log(np.exp(Z).sum(axis=1))
    true = Z[np.arange(len(y)), y]
    return float(np.mean(lse - true))


def fit_temperature(Xn, y, W, b, grid=None):
    """Fit a single temperature on held-out data (standard temperature scaling).

    The raw softmax was under-confident: it stated ~0.79 where accuracy was ~0.94.
    Dividing the logits by T<1 sharpens the distribution and aligns stated
    confidence with observed accuracy, which is what lets the agent trust a high
    score. Chosen by minimising held-out NLL - one scalar, no refit of weights.
    Note T only changes confidences, never the argmax, so accuracy is untouched.
    """
    if grid is None:
        grid = np.concatenate([np.arange(0.5, 3.01, 0.05)])
    best_t, best_nll = 1.0, float("inf")
    true = y
    for t in grid:
        Z = (Xn @ W + b) / t
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)
        nll = -np.log(np.clip(P[np.arange(len(y)), true], 1e-12, None)).mean()
        if nll < best_nll:
            best_nll, best_t = float(nll), float(t)
    return best_t, best_nll


def expected_calibration_error(probs, y, bins=10):
    top = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (top > lo) & (top <= hi)
        if m.sum():
            ece += m.mean() * abs(top[m].mean() - correct[m].mean())
    return float(ece)
