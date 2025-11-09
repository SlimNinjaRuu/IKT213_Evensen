"""
IKT213 – Assignment 3

"""

import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np



# Project paths / file names

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAMBO_FILE = "lambo.png"
SHAPES_FILE = "shapes-1.png"
TEMPLATE_FILE = "shapes_template.jpg"



# Small I/O helpers

def read_img(path: Path, flags=cv2.IMREAD_COLOR) -> np.ndarray:

    # Read an image with OpenCV and a clear error if missing.

    img = cv2.imread(str(path), flags)
    if img is None:
        raise FileNotFoundError(
            f"Could not load: {path}. Make sure the file exists and the name is correct."
        )
    return img


def save_img(img: np.ndarray, name: str) -> Path:

    # Save image into outputs folder and return the path.

    out_path = OUT_DIR / name
    cv2.imwrite(str(out_path), img)
    return out_path



# Assignment Part II functions
#Blur with Gaussian, converts grayscale and convert to unit8

def sobel_edge_detection(image: np.ndarray) -> np.ndarray:

    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY) if blurred.ndim == 3 else blurred
    sobel_64f = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=1)
    sobel_abs = np.absolute(sobel_64f)
    sobel_u8 = np.uint8(np.clip(sobel_abs, 0, 255))
    return sobel_u8


def canny_edge_detection(image: np.ndarray, threshold_1: int = 50, threshold_2: int = 50) -> np.ndarray:

    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY) if blurred.ndim == 3 else blurred
    edges = cv2.Canny(gray, threshold_1, threshold_2)
    return edges

# template matching with croos correlation

def template_match(image: np.ndarray, template: np.ndarray, threshold: float = 0.9) -> np.ndarray:

    # Prepare grayscale inputs
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    tmpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if template.ndim == 3 else template

    th, tw = tmpl_gray.shape[:2]

    # Run template matching
    score = cv2.matchTemplate(gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(score >= threshold)

    # Draw on a color copy for visibility
    draw = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for y, x in zip(ys, xs):
        cv2.rectangle(draw, (int(x), int(y)), (int(x + tw), int(y + th)), (0, 0, 255), 2)
    return draw

# pyramid resize
def resize(image: np.ndarray, scale_factor: int, up_or_down: str) -> np.ndarray:

    if scale_factor < 1:
        raise ValueError("scale_factor must be >= 1")

    out = image.copy()
    direction = up_or_down.lower()
    if direction == "up":
        for _ in range(scale_factor):
            out = cv2.pyrUp(out)     # doubles width & height each step
    elif direction == "down":
        for _ in range(scale_factor):
            out = cv2.pyrDown(out)   # halves width & height each step
    else:
        raise ValueError('up_or_down must be "up" or "down"')
    return out


# Script runner (produces deliverables)


def main() -> None:
    # Sobel & Canny on lambo.png
    lambo = read_img(DATA_DIR / LAMBO_FILE, cv2.IMREAD_COLOR)
    sobel_img = sobel_edge_detection(lambo)
    canny_img = canny_edge_detection(lambo, threshold_1=50, threshold_2=50)
    print("[Saved]", save_img(sobel_img, "lambo_sobel.png"))
    print("[Saved]", save_img(canny_img, "lambo_canny_50_50.png"))

    # Template matching on shapes-1.png with shapes_template.jpg
    shapes_img = read_img(DATA_DIR / SHAPES_FILE, cv2.IMREAD_COLOR)
    tmpl_img = read_img(DATA_DIR / TEMPLATE_FILE, cv2.IMREAD_COLOR)
    tm_annotated = template_match(shapes_img, tmpl_img, threshold=0.9)
    print("[Saved]", save_img(tm_annotated, "shapes_template_match.png"))

    # Resize up/down (image pyramids)
    up2 = resize(lambo, scale_factor=2, up_or_down="up")
    down2 = resize(lambo, scale_factor=2, up_or_down="down")
    print("[Saved]", save_img(up2, "lambo_up_x2.png"))
    print("[Saved]", save_img(down2, "lambo_down_x2.png"))

    print("\nDone → check assignment_3/outputs/")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
