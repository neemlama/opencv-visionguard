"""OpenCV 5 vision metrics — blur, glare, edges/defects. COOL accelerates these ops on Graviton."""
import cv2
import numpy as np

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
    edges = cv2.Canny(blurred, 50, 150)
    edge_density = float((edges > 0).mean())
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = sum(1 for c in contours if cv2.contourArea(c) > 500)

    # 4. Dark-blob anomalies (pits, porosity, corrosion) — a pit on a plate has
    # LOW edge density, so edges alone miss it. Adaptive threshold finds regions
    # darker than local surroundings; area filter removes harmless texture.
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 9
    )
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dcontours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        "edge_density": round(edge_density, 4),
        "large_contours": int(large_contours),
        "dark_area_ratio": round(dark_area_ratio, 5),
        "dark_blobs": int(dark_blobs),
        "defect_score": round(defect_score, 3),
        "needs_zoom": bool(needs_zoom),
        "image_shape": list(img.shape),
    }
