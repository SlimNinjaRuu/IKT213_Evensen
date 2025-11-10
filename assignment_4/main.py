import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# safe image loader, fails fast if the file cant be read
def read_or_die(path: Path, flags=cv2.IMREAD_COLOR) -> np.ndarray:
    img = cv2.imread(str(path), flags)
    if img is None:
        raise FileNotFoundError(f"Could not load: {path}")
    return img

# saves the images into output/
def save(img: np.ndarray, name: str) -> Path:
    p = OUT / name
    cv2.imwrite(str(p), img)
    print("[Saved]", p)
    return p

# Harris Corener Detection
def harris_corner_detection(
    image_bgr: np.ndarray,
    block_size: int = 2,
    ksize: int = 3,
    k: float = 0.04,
    thresh_ratio: float = 0.01,
) -> np.ndarray:

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_f32 = np.float32(gray)

    # Harris response
    dst = cv2.cornerHarris(gray_f32, block_size, ksize, k)
    dst = cv2.dilate(dst, None)

    out = image_bgr.copy()
    m = dst.max()
    if m > 0:
        out[dst > thresh_ratio * m] = [0, 0, 255]  # red marks (BGR)
    return out



# SIFT-based Image Alignment
def detect_and_describe_sift(gray: np.ndarray, nfeatures: int) -> Tuple[list, np.ndarray]:

    # Garyscale image
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    kps, desc = sift.detectAndCompute(gray, None)
    return kps, desc


def match_sift_flann(desc1: np.ndarray, desc2: np.ndarray) -> list:

    # KD-tree for SIFT and perform k=2 matching to enable Lowe's ratio
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE=1
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches_knn = flann.knnMatch(desc1, desc2, k=2)
    return matches_knn


# align image to align onto reference via SIFT + FLANN + Homography
def align_images_sift(
    image_to_align_bgr: np.ndarray,
    reference_bgr: np.ndarray,
    max_features: int = 10,
    lowe_ratio: float = 0.7
) -> Tuple[np.ndarray, np.ndarray]:

    g1 = cv2.cvtColor(image_to_align_bgr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)

    kp1, des1 = detect_and_describe_sift(g1, nfeatures=max_features)
    kp2, des2 = detect_and_describe_sift(g2, nfeatures=max_features)
    if des1 is None or des2 is None:
        raise RuntimeError("SIFT descriptors not found (try increasing max_features).")

    matches_knn = match_sift_flann(des1, des2)

    good = []
    for pair in matches_knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < lowe_ratio * n.distance:
            good.append(m)

    print(f"[SIFT] keypoints: {len(kp1)} vs {len(kp2)}, good matches: {len(good)}")
    if len(good) < 4:
        raise RuntimeError(f"Not enough good matches: {len(good)} (need >= 4). Try larger max_features or a looser ratio.")

    # build points from good matches
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Homography (RANSAC), estimate the homography with RANSAC to reject outliers
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC)
    if H is None:
        raise RuntimeError("Homography estimation failed.")

    # warp into the reference image plane
    h, w = reference_bgr.shape[:2]
    aligned = cv2.warpPerspective(image_to_align_bgr, H, (w, h))


    matches_mask = mask.ravel().tolist() if mask is not None else None
    matches_vis = cv2.drawMatches(
        image_to_align_bgr, kp1,
        reference_bgr,     kp2,
        good, None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        matchesMask=matches_mask,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return aligned, matches_vis



# PDF export
def export_pdf(pages: list[Path], name: str = "assignment4_output.pdf") -> Path:
    imgs = [Image.open(p).convert("RGB") for p in pages]
    pdf_path = OUT / name
    if not imgs:
        raise RuntimeError("No pages to export.")
    first, rest = imgs[0], imgs[1:]
    first.save(pdf_path, save_all=True, append_images=rest)
    print("[Saved PDF]", pdf_path)
    return pdf_path


def main():
    # Harris corners on the reference
    reference = read_or_die(DATA / "reference_img.png", cv2.IMREAD_COLOR)
    harris_img = harris_corner_detection(reference)
    p1 = save(harris_img, "harris.png")  # page 1

    # SIFT-based alignment
    to_align = read_or_die(DATA / "align_this.jpg", cv2.IMREAD_COLOR)

    try:
        aligned, matches = align_images_sift(
            image_to_align_bgr=to_align,
            reference_bgr=reference,
            max_features=10,
            lowe_ratio=0.7,
        )
    except Exception as e:
        print("[WARN] Strict SIFT settings failed:", e)

        aligned, matches = align_images_sift(
            image_to_align_bgr=to_align,
            reference_bgr=reference,
            max_features=1500,  # more features
            lowe_ratio=0.75,    # a touch looser
        )

    p2 = save(aligned, "aligned.png")   # page 2
    p3 = save(matches, "matches.png")   # page 3


    export_pdf([p1, p2, p3])


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"ERROR: {err}", file=sys.stderr)
        sys.exit(1)
