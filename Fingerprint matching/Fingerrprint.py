import os, time, csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ---------- Preprocessing tuned for fingerprints ----------
def preprocess_fingerprint(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise FileNotFoundError(path)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(g)
    g = cv2.GaussianBlur(g, (3,3), 0)
    _, binv = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binv

# ---------- One pair match ----------
def match_pair(p1, p2, method="ORB", ratio=0.7, crosscheck=False):
    i1, i2 = preprocess_fingerprint(p1), preprocess_fingerprint(p2)
    t0 = time.perf_counter()

    if method.upper() == "ORB":
        det = cv2.ORB_create(nfeatures=1500)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    elif method.upper() == "SIFT":
        det = cv2.SIFT_create(nfeatures=1500)
        index_params = dict(algorithm=1, trees=5)  # KDTree
        search_params = dict(checks=64)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        raise ValueError("method must be ORB or SIFT")

    kp1, d1 = det.detectAndCompute(i1, None)
    kp2, d2 = det.detectAndCompute(i2, None)

    if d1 is None or d2 is None or len(kp1) < 2 or len(kp2) < 2:
        return dict(good=0, kp1=len(kp1 or []), kp2=len(kp2 or []),
                    ms=1000*(time.perf_counter()-t0), vis=None)

    knn = matcher.knnMatch(d1, d2, k=2)
    good = [m for m, n in knn if m.distance < ratio * n.distance]

    if crosscheck and good:
        knn_ba = matcher.knnMatch(d2, d1, k=2)
        good_ba = [m for m, n in knn_ba if m.distance < ratio * n.distance]
        ba = {(m.trainIdx, m.queryIdx) for m in good_ba}
        good = [m for m in good if (m.queryIdx, m.trainIdx) in ba]

    vis = cv2.drawMatches(i1, kp1, i2, kp2, good, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return dict(good=len(good), kp1=len(kp1), kp2=len(kp2),
                ms=1000*(time.perf_counter()-t0), vis=vis)

# ---------- Evaluate your Data_check structure ----------
def process_data_check(root, out_dir, method="ORB", threshold=20, save_csv=True):
    if not os.path.isdir(root):
        raise FileNotFoundError(f"data_check folder not found: {root}")
    os.makedirs(out_dir, exist_ok=True)

    if save_csv:
        csv_path = os.path.join(out_dir, f"stats_{method.lower()}.csv")
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["pair_folder","method","kp1","kp2","good_matches","runtime_ms","pred(1=same)","actual(1=same)"]
            )

    y_true, y_pred = [], []

    for pair_folder in sorted(os.listdir(root)):
        pf_path = os.path.join(root, pair_folder)
        if not os.path.isdir(pf_path):
            continue

        # label from folder name: "same_*" -> 1, "different_*" -> 0
        name = pair_folder.lower()
        if name.startswith("same"):
            actual = 1
        elif name.startswith("different"):
            actual = 0
        else:
            print(f"Skipping {pair_folder}: name should start with 'same' or 'different'")
            continue

        imgs = sorted([f for f in os.listdir(pf_path)
                       if f.lower().endswith((".tif",".tiff",".png",".jpg",".jpeg"))])
        if len(imgs) != 2:
            print(f"Skipping {pair_folder}: expected 2 images, found {len(imgs)}")
            continue

        p1, p2 = os.path.join(pf_path, imgs[0]), os.path.join(pf_path, imgs[1])
        res = match_pair(p1, p2, method=method, ratio=0.7, crosscheck=False)

        pred = 1 if res["good"] >= threshold else 0
        y_true.append(actual); y_pred.append(pred)

        tag = "matched" if pred == 1 else "unmatched"
        print(f"{pair_folder}: {tag.upper()} | matches={res['good']} "
              f"(kp {res['kp1']}/{res['kp2']}, {res['ms']:.1f} ms, {method})")

        if res["vis"] is not None:
            cv2.imwrite(os.path.join(out_dir, f"{pair_folder}_{method.lower()}_{tag}.png"), res["vis"])

        if save_csv:
            with open(os.path.join(out_dir, f"stats_{method.lower()}.csv"), "a", newline="") as f:
                csv.writer(f).writerow([pair_folder, method, res["kp1"], res["kp2"],
                                        res["good"], f"{res['ms']:.2f}", pred, actual])

    # ---- Confusion matrix plot ----
    plt.close('all')  # <<< clears any stray figures before plotting
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Different (0)", "Same (1)"])
    plt.figure(figsize=(5.5,5.5))
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix – {method} (threshold={threshold})")
    plt.tight_layout()
    plt.show()

# ---------- Set YOUR paths (macOS) ----------
data_check_root = "/Users/michaelevensen/IKT213 machine vision/IKT213_Evensen/Fingerprint matching/Data_check"
out_dir         = "/Users/michaelevensen/IKT213 machine vision/IKT213_Evensen/Fingerprint matching/Data_check_19"

# Fast baseline (your best was threshold=19)
process_data_check(data_check_root, out_dir, method="ORB", threshold=19)


