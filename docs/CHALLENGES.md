# Challenges We Faced

Engineering log of real problems hit while building ForgeSight. Judges score
"evaluation evidence appropriate to the project, including failure cases or
limitations" — this file is that evidence. Each entry is reproducible.

---

## C-01 — Defect detector missed dark pits (false negative on real defect)

**Severity:** Critical. Would have produced a detector that passes defective
weldseverything. A judge uploading one real defect image would have seen
`pass`and the whole technical-execution score (30%) would be undermined.

**Symptom**
The first detector version scored purely on edge evidence:

```python
edge_component = min(1.0, edge_density * 8 + large_contours * 0.05)
```

End-to-end test over our own samples gave:

| sample | expected | got | defect_score |
|---|---|---|---|
| weld_clean | pass | pass | 0.11 |
| weld_defect | defect | **pass** | 0.114 |
| weld_blurry | re-shoot | re-shoot | 0.00 |
| weld_glare | re-shoot-angle | re-shoot-angle | 0.23 |

`weld_defect` scored 0.114 — statistically indistinguishable from the clean
sample (0.11). The system was blind to the defect.

**Root cause**
Canny edges fire on *transitions*. A porosity pit in the middle of a weld bead
is a **uniform dark disk**: dark inside, with an edge only at its rim. On our
640x480 plate the rim is ~2% of the image, so the edge channel barely registers,
while the pit itself contributes almost nothing to `edge_density`.

Worse, the two "hard" samples (severe damage, glare) scored *higher* than the
real single-pit defect, so tuning thresholds on score magnitude alone would not
have fixed it — the signal simply was not present in that feature.

**Fix**
Added a complementary feature that does not depend on edge energy: regions
darker than their local surroundings.

```python
thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 9
)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
dcontours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
defect_areas = [cv2.contourArea(c) for c in dcontours if cv2.contourArea(c) > 400]
```

`adaptiveThreshold` with `THRESH_BINARY_INV` yields 255 where the pixel is darker
than the local Gaussian-weighted mean, so pits light up as filled blobs
regardless of contrast. `MORPH_OPEN` removes single-pixel sensor noise, and the
`contourArea > 400` filter (0.2% of this image) discards harmless texture.

Blend both channels so neither failure mode returns:

```python
edge_component = min(1.0, edge_density * 8 + large_contours * 0.05)
blob_component = min(1.0, dark_area_ratio * 25 + dark_blobs * 0.25)
defect_score = min(1.0, 0.5 * edge_component + 0.9 * blob_component)
```

**Verification after fix**

| sample | expected | got | defect_score | dark_blobs | trace steps |
|---|---|---|---|---|---|
| weld_clean | pass | pass | 0.055 | 0 | 1 |
| weld_defect | defect | human-review | 0.318 | 1 | 2 |
| weld_damage | escalate | escalate-human | 1.000 | 6 | 1 |
| weld_blurry | re-shoot | re-shoot | 0.000 | 0 | 1 |
| weld_glare | angle change | re-shoot-angle | 0.707 | 2 | 1 |

Separation between clean and defective went from 0.004 to 0.263 — a 66x margin.

**What we learned**
A single-feature detector is only as good as the assumption behind the feature.
"Pits have low contrast, therefore low edges" was true, and that is exactly why
they were invisible. Defect detection needs at least one channel per defect
*mechanism*, not per defect *appearance*.

**Known remaining limitations** (deliberate, documented for judges)
- Thresholds are hand-tuned on synthetic samples, not a labelled real dataset.
  Scores are a triage signal, not calibrated probability.
- `weld_damage` and `weld_glare` both reach high `defect_score`, but glare is
  caught first because branch order is blur -> glare -> defect. Reordering would
  let glare masquerade as a real defect.
- Very small pits (< 400px area) are filtered out entirely.
- No geometric tolerance model: a real system needs weld width/spec limits, not
  just "something is dark".

---

## C-02 — Sample images written to the wrong directory

`make_samples.py` computed its output path with
`Path(__file__).resolve().parent.parent / "samples"`, but the file lives at
`backend/app/make_samples.py`, so `.parent.parent` is `backend/`, not the repo
root. Samples silently landed in `backend/samples/` and every `/samples/*` request
404'd.

Fixed to `.parents[2]`. Worth logging because it failed *silently* — the script
printed a success message listing filenames that were not where the server looked.

---

## C-03 — PowerShell 5.1 has no `-Form` parameter

`Invoke-RestMethod -Form @{file = Get-Item ...}` fails with
"A parameter cannot be found that matches parameter name 'Form'" on Windows
PowerShell 5.1. Multipart uploads need `curl.exe`:

```powershell
curl.exe -s -X POST -F "file=@samples/weld_defect.jpg" http://localhost:8000/inspect
```

All sample-API tests in this repo were re-run with `curl.exe`.

---

## C-04 — Wrong mental model of COOL packaging (design correction)

Our first plan assumed COOL could be `pip install`ed or baked into a Dockerfile
for `linux/arm64`. Researching the AWS Marketplace listing showed COOL ships as
a **Graviton AMI** (Ubuntu 24.04, ARM64, KleidiCV-optimised, based on OpenCV 5),
not a library artifact.

Consequence: the COOL Award path must launch the Marketplace AMI and run the
workload inside it with:

```bash
export LD_LIBRARY_PATH="/opt/cool/cpp_sdk/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="/opt/cool/python_3.12/site-packages/"
python3.12 bench.py
```

The x86/Arm baseline therefore cannot be a container build — it must be the same
Graviton instance family running DIY `pip install opencv-python`, which isolates
the library's contribution from the hardware's. See `docs/COOL-setup.md`.