import cv2
import numpy as np



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


def crop(image, x_0, x_1, y_0, y_1):
    return cropping(image, x_0, x_1, y_0, y_1)

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
            # B, G, R
            empty_image[y, x, 0] = image[y, x, 0]
            empty_image[y, x, 1] = image[y, x, 1]
            empty_image[y, x, 2] = image[y, x, 2]
    return empty_image

def grayscale(image):

    if image is None:
        raise ValueError("image cannot be none")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def hsv(image):

    if image is None:
        raise ValueError("image cannot be None")
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


hvs = hsv

def hue_shifted(image, emptyPictureArray, hue=50):

    if image is None:
        raise ValueError("image is no where to be found")
    if emptyPictureArray.shape != image.shape or emptyPictureArray.dtype != np.uint8:
        raise ValueError("emptyPictureArray must be shape (h, w, 3) and dtype uint8")

    # BGR -> RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # +hue to all channels with wrap-around
    shifted_rgb = (rgb.astype(np.int16) + int(hue)) % 256
    shifted_rgb = shifted_rgb.astype(np.uint8)

    # RGB -> BGR
    shifted_bgr = cv2.cvtColor(shifted_rgb, cv2.COLOR_RGB2BGR)

    np.copyto(emptyPictureArray, shifted_bgr)
    return emptyPictureArray

def smoothing(image):

    if image is None:
        raise ValueError("image is no where to be found")
    return cv2.GaussianBlur(image, (15, 15), 0, borderType=cv2.BORDER_DEFAULT)

def rotation(image, rotation_angle):

    if image is None:
        raise ValueError("image is no where to be found")
    if rotation_angle == 180:
        rotated = cv2.rotate(image, cv2.ROTATE_180)
    elif rotation_angle == 90:
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    else:
        raise ValueError("Invalid rotation angle (use 90 or 180)")
    return rotated


image = cv2.imread("data/lena.png", cv2.IMREAD_COLOR)
if image is None:
    raise FileNotFoundError("Could not find data/lena.png")
h, w = image.shape[:2]
if (h, w) != (512, 512):
    raise ValueError(f"Lena must be 512x512; got {w}x{h}")

px = image[100, 100]
print("px @ (100,100):", px)
print("shape:", image.shape, "size:", image.size, "dtype:", image.dtype)



# Padding
padded = padding(image, 100)

# Crop face
cropped = cropping(image, 80, w - 130, 80, h - 130)

# Resize to 200x200
resized = resize_image(image, 200, 200)

#  Manual copy
empty_image = np.zeros((h, w, 3), dtype=np.uint8)
manual_copy_img = copy(image, empty_image)

# Grayscale
gray = grayscale(image)

#HSV
hsv_image = hsv(image)
preview_bgr = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

# Color shift
empty = np.zeros_like(image, dtype=np.uint8)
shifted = hue_shifted(image, empty, hue=50)

# Smoothing (15x15)
smoothed = smoothing(image)

#Rotations
rot180 = rotation(image, 180)
rot90  = rotation(image, 90)


# press q to quit the display

while True:
    cv2.imshow("reflect (padded)", padded)
    cv2.imshow("cropped", cropped)
    cv2.imshow("resized 200x200", resized)
    cv2.imshow("manual_copy", manual_copy_img)
    cv2.imshow("grayscale", gray)
    cv2.imshow("HSV preview (BGR space)", preview_bgr)
    cv2.imshow("Original", image)
    cv2.imshow("Hue shifted (+50 in RGB)", shifted)
    cv2.imshow("smoothed (15x15)", smoothed)
    cv2.imshow("rot180", rot180)
    cv2.imshow("rot90", rot90)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break
cv2.destroyAllWindows()

# saves the images

cv2.imwrite("out_padded.png", padded)
cv2.imwrite("out_cropped.png", cropped)
cv2.imwrite("out_resized_200x200.png", resized)
cv2.imwrite("out_manual_copy.png", manual_copy_img)
cv2.imwrite("out_gray.png", gray)
cv2.imwrite("out_hsv.png", hsv_image)
cv2.imwrite("out_preview.png", preview_bgr)
cv2.imwrite("out_hue_shifted.png", shifted)
cv2.imwrite("out_smoothed.png", smoothed)
cv2.imwrite("out_rot180.png", rot180)
cv2.imwrite("out_rot90.png", rot90)
