import cv2
import numpy as np

# func

def padding(image, border_width=100):

    return cv2.copyMakeBorder(
        image, border_width, border_width, border_width, border_width,
        cv2.BORDER_REFLECT
    )

def cropping(image, x_0, x_1, y_0, y_1):

    h, w = image.shape[:2]
    x0 = max(0, int(x_0)); x1 = min(w, int(x_1))
    y0 = max(0, int(y_0)); y1 = min(h, int(y_1))
    if x0 >= x1 or y0 >= y1:
        raise ValueError("Invalid crop box after clamping")
    return image[y0:y1, x0:x1].copy()

def resize_image(image, width, height, interp=cv2.INTER_LINEAR):

    return cv2.resize(image, (int(width), int(height)), interpolation=interp)

def copy(image, empty_image):

    if image is None:
        raise ValueError("image cannot be None")
    h, w = image.shape[:2]
    if empty_image.shape != (h, w, 3) or empty_image.dtype != np.uint8:
        raise ValueError("empty_image must be shape (h, w, 3) and dtype uint8")

    for y in range(h):
        for x in range(w):
            # B, G, R channels
            empty_image[y, x, 0] = image[y, x, 0]
            empty_image[y, x, 1] = image[y, x, 1]
            empty_image[y, x, 2] = image[y, x, 2]
    return empty_image

def grayscale(image):

    if image is None:
        raise ValueError("image cannot be None")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)




def hvs(image):
    if image is None:
        raise ValueError("image cannot be None")

    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#load color image
image = cv2.imread("lena-1.png", cv2.IMREAD_COLOR)
if image is None:
    raise FileNotFoundError("No image")


hvs_image = hvs(image)
cv2.imwrite("out_HSV.png", hvs_image)

preview_bgr = cv2.cvtColor(hvs_image, cv2.COLOR_HSV2BGR)

cv2.imshow('Original (BGR)', image)
cv2.imshow('HSV image', preview_bgr)



def hue_shifted(image, emptyPictureArray, hue=50):
    if image is None:
        raise ValueError("image is no where to be found")
    if emptyPictureArray.shape != image.shape or emptyPictureArray.dtype != np.uint8:
        raise ValueError("emptyPictureArray must be shape (h, w, 3) and dtype uint8")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    h = (h.astype(np.int16) + hue) % 256 # limit
    h = h.astype(np.uint8)

    hsv_shifted = cv2.merge((h, s, v))

    shifted_bgr = cv2.cvtColor(hsv_shifted, cv2.COLOR_HSV2BGR)

    np.copyto(emptyPictureArray, shifted_bgr)

    return emptyPictureArray

def smoothing(image):
    if image is None:
        raise ValueError("image is no where to be found")

    blurred = cv2.GaussianBlur(image, (11, 11), 0, borderType=cv2.BORDER_DEFAULT)
    return blurred


def rotation(image, rotation_angle):
    if image is None:
        raise ValueError("image is no where to be found")

    if rotation_angle == 180:
        rotated = cv2.rotate(image, cv2.ROTATE_180)
    elif rotation_angle == 90:
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    else:
        raise ValueError("Invalid rotation angle")

    return rotated


img = cv2.imread("lena-1.png", cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError("Couldn't find lena-1.png")

px = img[100, 100]
print("px @ (100,100):", px)
print("shape:", img.shape, "size:", img.size, "dtype:", img.dtype)

padded = padding(img, 100)

h, w = img.shape[:2]
cropped = cropping(img, 80, w - 130, 80, h - 130)

resized = resize_image(img, 200, 200)

empty_image = np.zeros((h, w, 3), dtype=np.uint8)
manual_copy = copy(img, empty_image)

gray = grayscale(img)

empty = np.zeros_like(image, dtype=np.uint8)
shifted = hue_shifted(image, empty, hue=50)

smoothed = smoothing(image)

rot180 = rotation(image, 180)
rot90 = rotation(image, 90)


# Display; press 'q' to quit
while True:
    cv2.imshow("reflect", padded)
    cv2.imshow("cropped", cropped)
    cv2.imshow("resized 200x200", resized)
    cv2.imshow("manual_copy", manual_copy)
    cv2.imshow("grayscale", gray)

    cv2.imshow("hvs image", hvs_image)
    cv2.imshow("preview", preview_bgr)

    cv2.imshow("Original", img)
    cv2.imshow("Hue shifted (+50)", shifted)

    cv2.imshow("smoothed", smoothed)

    cv2.imshow("rot180", rot180)
    cv2.imshow("rot90", rot90)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break
cv2.destroyAllWindows()

# Save to NEW filenames so you don't overwrite your source
cv2.imwrite("out_padded.png", padded)
cv2.imwrite("out_cropped.png", cropped)
cv2.imwrite("out_resized_200x200.png", resized)
cv2.imwrite("out_manual_copy.png", manual_copy)
cv2.imwrite("out_gray.png", gray)
cv2.imwrite("out_hvs.png", hvs_image)
cv2.imwrite("out_preview.png", preview_bgr)
cv2.imwrite("hue_shifted.png", shifted)
cv2.imwrite("out_hue_shifted.png", shifted)
cv2.imwrite("out_smoothed.png", smoothed)
cv2.imwrite("out_rot180.png", rot180)
cv2.imwrite("out_rot90.png", rot90)