"""Capture simulator driven by camera SETTINGS, not pixel patches.

The first version applied fixes directly to pixels (an unsharp mask to undo
Gaussian blur). That does not converge: sharpening cannot recreate
high-frequency detail that the blur destroyed, so the loop asked for a refocus
forever and never satisfied the quality gate.

Physically, a blurry photo is not a blurry scene - it is a sharp scene captured
with the wrong settings. So the simulator holds the scene and a set of capture
settings, and a "fix" changes the settings and re-renders. That converges in one
step, and it is the honest model: the scene does not change, the camera does.

It remains a simulator, not hardware, and the report says so. What it proves is
that each agent request is action-producing and that the loop terminates.

Degradations and fixes are applied to real NEU photographs, so no part of the
evaluation depends on generated imagery.
"""
import base64

import cv2
import numpy as np

CAPTURE_SETTINGS = ("focus_blur", "exposure", "glare", "noise", "contrast")

FIXES = {
    "steady_and_refocus": "tripod + autofocus lock, then recapture",
    "raise_exposure": "increase illumination and exposure, then recapture",
    "change_angle_or_polarizer": "tilt 15 degrees / fit polariser, then recapture",
    "boost_contrast": "increase local contrast, then recapture",
    "denoise": "enable denoising, then recapture",
}

# Which setting each fix corrects.
FIX_EFFECT = {
    "steady_and_refocus": ("focus_blur", 0.0),
    "raise_exposure": ("exposure", 1.0),
    "change_angle_or_polarizer": ("glare", 0.0),
    "boost_contrast": ("contrast", 1.0),
    "denoise": ("noise", 0.0),
}


class CaptureSimulator:
    """Holds a scene plus capture settings; re-renders on demand.

    settings:
        focus_blur  sigma of defocus blur (0 = sharp)
        exposure    multiplicative gain (1.0 = correct)
        glare       radius fraction of a specular highlight disc (0 = none)
        noise       Gaussian noise sigma in grey levels
        contrast    multiplier on contrast about mid-grey (1.0 = correct)
    """

    def __init__(self, scene, focus_blur=0.0, exposure=1.0, glare=0.0,
                 noise=0.0, contrast=1.0, seed=7):
        self.scene = scene
        self.settings = {
            "focus_blur": float(focus_blur),
            "exposure": float(exposure),
            "glare": float(glare),
            "noise": float(noise),
            "contrast": float(contrast),
        }
        self.seed = seed
        self.log = []

    # ---------- rendering ----------
    def render(self):
        img = self.scene.copy()
        s = self.settings
        if s["contrast"] != 1.0:
            img = np.clip((img.astype(np.float32) - 128) * s["contrast"] + 128,
                          0, 255).astype(np.uint8)
        if s["exposure"] != 1.0:
            img = np.clip(img.astype(np.float32) * s["exposure"], 0, 255).astype(np.uint8)
        if s["glare"] > 0:
            h, w = img.shape[:2]
            r = int(min(h, w) * s["glare"])
            out = img.copy()
            cv2.circle(out, (w // 2, h // 2), r, (255, 255, 255), -1)
            img = cv2.addWeighted(out, 0.55, np.full_like(out, 255), 0.45, 0)
        if s["focus_blur"] > 0:
            img = cv2.GaussianBlur(img, (0, 0), s["focus_blur"])
        if s["noise"] > 0:
            rng = np.random.default_rng(self.seed)
            img = np.clip(img.astype(np.float32)
                          + rng.normal(0, s["noise"], img.shape), 0, 255).astype(np.uint8)
        return img

    @property
    def image(self):
        return self.render()

    # ---------- action ----------
    def apply(self, fix):
        """Correct one setting. Returns the newly rendered capture."""
        if fix not in FIX_EFFECT:
            raise ValueError(f"unknown fix {fix}")
        key, target = FIX_EFFECT[fix]
        before = self.settings[key]
        self.settings[key] = target
        self.log.append({"fix": fix, "setting": key, "before": before, "after": target})
        return self.render()

    @property
    def degraded(self):
        s = self.settings
        return any(abs(s[k] - d) > 1e-9 for k, d in
                   (("focus_blur", 0.0), ("exposure", 1.0), ("glare", 0.0),
                    ("noise", 0.0), ("contrast", 1.0)))


# Named starting conditions, so a demo is reproducible.
PRESETS = {
    "soft_focus": {"focus_blur": 4.0},
    "under_exposed": {"exposure": 0.70},
    "specular_glare": {"glare": 0.30},
    "low_contrast": {"contrast": 0.60},
    "sensor_noise": {"noise": 18.0},
    "clean": {},
}
DEGRADATIONS = {
    "soft_focus": "defocus, sigma 4.0",
    "under_exposed": "30% exposure loss",
    "specular_glare": "specular highlight disc, r=30% of frame",
    "low_contrast": "contrast crushed to 60%",
    "sensor_noise": "Gaussian noise sigma 18",
    "clean": "no degradation",
}


def make(scene, preset="clean", **overrides):
    return CaptureSimulator(scene, **PRESETS.get(preset, {}), **overrides)


def b64_jpeg(image, quality=85):
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")
