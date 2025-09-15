import cv2
import os

def print_image_information(image):

    h, w = image.shape[:2]
    c = image.shape[2] if len(image.shape) == 3 else 1
    print("Image height:", h)
    print("Image width:", w)
    print("Image channels:", c)
    print("Image size (number of values):", image.size)
    print("Image data type:", image.dtype)

def save_camera_information():

    cap = cv2.VideoCapture(0)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_dir = os.path.expanduser("~/IKT213_MacBook_Pro_M1/assignment_1/solutions")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "camera_outputs.txt"), "w") as f:
        f.write(f"fps: {fps}\n")
        f.write(f"height: {height}\n")
        f.write(f"width: {width}\n")

    print("Camera information saved!")

    cap.release()

def main():

    img = cv2.imread("lena-1.png")
    if img is not None:
        print_image_information(img)


    save_camera_information()

if __name__ == "__main__":
    main()
