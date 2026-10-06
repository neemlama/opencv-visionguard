"""OpenCV 5 vision metrics — blur, glare, edges/defects. COOL accelerates these ops on Graviton."""
import cv2
import numpy as np

# Black-hat dark-blob detector tuning, chosen by measurement in
# bench/detector_variants.py. BOX=25 keeps the rolling-ball scale ~1.4x the weld
# bead width while staying cheap (box filter is O(1) per pixel); MIN_CONTRAST
# rejects sensor noise; the 400px area floor discards harmless texture.
DARK_BLOB_BOX = 25
DARK_BLOB_MIN_CONTRAST = 18
# Canny thresholds are set from the image median. A full np.median over 640x480
# costs 2.019 ms and became the pipeline bottleneck on its own; sampling every 8th
# pixel costs 0.030 ms (67x faster) and returned the identical median on every
# sample tested (140.0 -> same 0.66/1.33 thresholds). Estimating a robust centre
# from 1/64 of the pixels is plenty.
MEDIAN_SAMPLE_STEP = 8

def analyze_image(data: bytes) -> dict:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "invalid image", "decision_hint": "re-shoot"}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Blur — Laplacian variance (COOL-accelerated)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_blurry = blur_score < 100.0

    # 2. Glare — % of very bright pixels in HSV V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    glare_ratio = float((v > 240).mean())
    has_glare = glare_ratio > 0.05

    # 3. Edges — Canny + contour count + mean edge strength
    # COOL path uses same calls; benchmark measures these.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Canny thresholds proportional to the image median, not fixed 50/150.
    # Absolute thresholds are illumination-dependent: at 30% lower light the
    # edge channel collapsed (edge_density 0.0079 -> 0.0025), which dragged the
    # blended score under the decision cut and auto-passed a real defect.
    # 0.66/1.33 of the median is the ratio OpenCV recommends when thresholds are
    # left unspecified.
    med = float(np.median(gray[::MEDIAN_SAMPLE_STEP, ::MEDIAN_SAMPLE_STEP]))
    canny_lo = max(1, int(0.66 * med))
    canny_hi = max(2, int(1.33 * med))
    edges = cv2.Canny(blurred, canny_lo, canny_hi)
    edge_density = float((edges > 0).mean())
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = sum(1 for c in contours if cv2.contourArea(c) > 500)

    # 4. Dark-blob anomalies (pits, porosity, corrosion) - a pit on a plate has
    # LOW edge density, so edges alone miss it (see docs/CHALLENGES.md C-01).
    #
    # Implemented as a black-hat (rolling-ball) transform using a BOX filter, not
    # adaptiveThreshold and not a large-kernel Gaussian. Both were measured
    # (bench/detector_variants.py):
    #
    #     variant                  ms/call   clean defect damage glare
    #     adaptiveThreshold(31,9)     2.428      0      1      6      2
    #     gaussTophat sigma=12        5.291      0      1      6      0
    #     boxTophat k=25              0.920      0      1      6      0   <- chosen
    #
    # cv2.blur is a box filter: O(1) per pixel regardless of kernel size, so it
    # stays cheap where a 25x25 Gaussian (51x51 implied kernel) costs 5.3ms.
    # Same recall as adaptiveThreshold, 2.6x faster, and it rejects the glare
    # sample that adaptiveThreshold flagged. This matters beyond speed:
    # adaptiveThreshold was ~74% of pipeline time and was the one operation COOL
    # measurably regressed on, so removing it is what makes the COOL arm viable.
    bg = cv2.blur(gray, (DARK_BLOB_BOX, DARK_BLOB_BOX))
    tophat = cv2.subtract(bg, gray)
    _, mask = cv2.threshold(tophat, DARK_BLOB_MIN_CONTRAST, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dcontours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    defect_areas = [cv2.contourArea(c) for c in dcontours if cv2.contourArea(c) > 400]
    dark_area_ratio = float(sum(defect_areas) / gray.size)
    dark_blobs = len(defect_areas)

    # Defect score blends edge evidence and dark-blob evidence so both
    # high-contrast cracks and low-contrast pits are caught.
    edge_component = min(1.0, edge_density * 8 + large_contours * 0.05)
    blob_component = min(1.0, dark_area_ratio * 25 + dark_blobs * 0.25)
    defect_score = float(min(1.0, 0.5 * edge_component + 0.9 * blob_component))
    needs_zoom = 0.3 < defect_score < 0.7 and not is_blurry and not has_glare

    return {
        "opencv_version": cv2.__version__,
        "blur_variance": round(blur_score, 2),
        "is_blurry": bool(is_blurry),
        "glare_ratio": round(glare_ratio, 4),
        "has_glare": bool(has_glare),
        "canny_thresholds": [canny_lo, canny_hi],
        "edge_density": round(edge_density, 4),
        "large_contours": int(large_contours),
        "dark_area_ratio": round(dark_area_ratio, 5),
        "dark_blobs": int(dark_blobs),
        "defect_score": round(defect_score, 3),
        "needs_zoom": bool(needs_zoom),
        "image_shape": list(img.shape),
    }
