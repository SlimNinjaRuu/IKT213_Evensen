import numpy as np
import cv2

# Reads the image information
img = cv2.imread('lena-1.png',1)

def print_image_informations(img):
    print("Image height: ", img.shape[0])
    print("Image width: ", img.shape[1])
    print("Image channels ", img.shape[2] if len(img.shape) == 3 else 1)
    print("Image data type: ", img.dtype)
    print("Image size:", img.size, "bytes")

    if len (img.shape) == 3:
        print("Blue channel sample:", img[0:3, 0:3, 0])
        print ("Green channel sample:", img[0:3, 0:3, 1])
        print ("Red channel sample:", img[0:3, 0:3, 2])


print_image_informations(img)



video_capture = cv2.VideoCapture(0)

frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video_capture.get(cv2.CAP_PROP_FPS)

output_file = "camera_output.txt"

with open(output_file, "w") as f:
    f.write("Frame width: " + str(frame_width) + "\n")
    f.write("Frame height: " + str(frame_height) + "\n")
    f.write("FPS: " + str(fps) + "\n")