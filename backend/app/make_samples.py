"""Generate synthetic inspection samples: clean weld, defect, blurry, glare."""
import cv2
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "samples"
OUT.mkdir(exist_ok=True)


def base_plate(w=640, h=480):
    img = np.full((h, w, 3), 140, np.uint8)
    noise = np.random.randint(-12, 12, (h, w, 1), dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def add_weld(img, y, defect=False):
    row = slice(y, y + 26)
    cv2.rectangle(img, (40, row.start), (600, row.stop), (198, 198, 198), -1)
    for x in range(40, 600, 7):
        cv2.circle(img, (x, y + 13), 5, (170, 170, 170), -1)
    if defect:
        cv2.circle(img, (320, y + 13), 13, (38, 38, 38), -1)
        cv2.circle(img, (320, y + 13), 6, (10, 10, 10), -1)
    return img


clean = add_weld(base_plate(), 120, defect=False)
cv2.imwrite(str(OUT / "weld_clean.jpg"), clean)

defect = add_weld(base_plate(), 120, defect=True)
for i in range(14):
    x = 250 + i * 22
    cv2.circle(defect, (x, 300 + (i % 3) * 9), 7, (60, 60, 60), -1)
cv2.imwrite(str(OUT / "weld_defect.jpg"), defect)

blurry = cv2.GaussianBlur(clean, (0, 0), 7.0)
cv2.imwrite(str(OUT / "weld_blurry.jpg"), blurry)

glare = clean.copy()
cv2.circle(glare, (300, 240), 150, (255, 255, 255), -1)
glare = cv2.addWeighted(glare, 0.55, np.full_like(glare, 255), 0.45, 0)
cv2.imwrite(str(OUT / "weld_glare.jpg"), glare)

# Severe porosity / undercut — drives defect_score high enough to escalate
damage = add_weld(base_plate(), 120, defect=True)
for i in range(26):
    x = 120 + i * 18
    cv2.circle(damage, (x, 130 + (i % 5) * 5), 10, (25, 25, 25), -1)
for i in range(12):
    x = 300 + i * 20
    cv2.ellipse(damage, (x, 340), (16, 9), 0, 0, 360, (45, 45, 45), -1)
cv2.imwrite(str(OUT / "weld_damage.jpg"), damage)

print("wrote:", ", ".join(p.name for p in sorted(OUT.glob("*.jpg"))))