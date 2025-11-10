# Assignment 4 – Feature Detection & Image Alignment (IKT213)

This assignment implements two computer vision tasks:

---

## 1) Harris Corner Detection
We detect Harris corners on the reference image (`reference_img.png`), draw detected corner pixels in **red**, and save the result.

Output page 1 (PDF):
- `harris.png`

Reference used:
https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html

---

## 2) Feature-Based Image Alignment (SIFT)
We use SIFT + FLANN + Lowe’s ratio test + RANSAC homography to align `align_this.jpg` to the same perspective as `reference_img.png`.

Parameters (spec):
- `max_features = 10`
- `good_match_percent = 0.7`

If alignment fails with strict spec parameters, the script retries with:
- more features
- slightly looser ratio
This makes the solution robust while still following the required approach.

Output page 2 (PDF):
- `aligned.png` (warped/registered image)

Output page 3 (PDF):
- `matches.png` (visualization of inlier matches)

Reference used:
https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html

---

