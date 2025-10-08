import os, time
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Preprocessing for natural images
def preprocess_uia(path, max_side=1400):
    img = cv2.imread(path, cv2.IMREAD_COLOR) # Read image at path as grayscale.
    if img is None:
        raise FileNotFoundError(path) # crash early if the path is wrong or image can’t load.
    h, w = img.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w)) # Compute a downscale factor so the longest side is at most
    if scale < 1.0:
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # grayscale


    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    return img, gray

# ORB + BF(Hamming) + Lowe + RANSAC
def match_uia(p1, p2, method="ORB", ratio=0.75): # matches features between two images using ORB
    img1, g1 = preprocess_uia(p1)
    img2, g2 = preprocess_uia(p2)
    t0 = time.perf_counter()

    if method.upper() == "ORB":
        det = cv2.ORB_create(nfeatures=4000, scaleFactor=1.2, nlevels=8) # build the detector/descriptor with up to 1500 features.
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    elif method.upper() == "SIFT":
        det = cv2.SIFT_create(nfeatures=4000) # build SIFT detector.
        index_params = dict(algorithm=1, trees=5)  # KDTree
        search_params = dict(checks=64)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        raise ValueError("method must be 'ORB' or 'SIFT'")

 # Detect keypoints and compute descriptors for both images.
    kp1, d1 = det.detectAndCompute(g1, None)
    kp2, d2 = det.detectAndCompute(g2, None)

    if d1 is None or d2 is None or len(kp1) < 2 or len(kp2) < 2:
        return dict(
            good=0, inliers=0, inlier_ratio=0.0, ms=1000*(time.perf_counter()-t0),
            vis=None, kp1=len(kp1 or []), kp2=len(kp2 or [])
        )

    knn = matcher.knnMatch(d1, d2, k=2)
    good = [m for m, n in knn if m.distance < ratio * n.distance]

    # RANSAC homography on good matches
    inliers = 0
    mask = None
    if len(good) >= 4:
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        if mask is not None:
            inliers = int(mask.ravel().sum())

    inlier_ratio = (inliers / max(len(good), 1)) if good else 0.0

    draw_params = dict(matchesMask=mask.ravel().tolist() if mask is not None else None,
                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    vis = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    return dict(
        good=len(good), inliers=inliers, inlier_ratio=float(inlier_ratio),
        ms=1000*(time.perf_counter()-t0), vis=vis, kp1=len(kp1), kp2=len(kp2)
    )

if __name__ == "__main__":
    # The UiA images from the assignment
    imgA = "UiA front1.png"
    imgB = "UiA front3.jpg"
    # paths to the two UiA images
    out_dir = "UiA_results"
    os.makedirs(out_dir, exist_ok=True)

    # Run ORB pipeline or SIFT (Swap between them on method)
    res = match_uia(imgA, imgB, method="SIFT", ratio=0.75)

    # Decision rule
    MATCH_THRESH_GOOD = 25
    MATCH_THRESH_INLIERS = 10
    verdict = "MATCH" if (res["good"] >= MATCH_THRESH_GOOD and res["inliers"] >= MATCH_THRESH_INLIERS) else "NO MATCH"

    print(f"Method: ORB")
    print(f"Keypoints: {res['kp1']} / {res['kp2']}")
    print(f"Good matches: {res['good']}")
    print(f"Inliers (RANSAC): {res['inliers']}  | Inlier ratio: {res['inlier_ratio']:.3f}")
    print(f"Time: {res['ms']:.1f} ms")
    print(f"Verdict: {verdict}")

    # Save and show visualization
    out_path = os.path.join(out_dir, f"uia_match_orb_{verdict.lower()}.png")
    if res["vis"] is not None:
        cv2.imwrite(out_path, res["vis"])
        print(f"Saved match visualization: {out_path}")