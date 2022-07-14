import argparse
import argparse
import numpy as np
import cv2
import pytesseract

def text_detect(frame):
    # takes frames, finds text and appended to txt
    # crop to only top and bottom of frame containing text, and convert to grayscale
    crops = crop_frame(frame)

    # store final processed frames
    processed_frames = []
    # apply all stages to each frame
    for crop in crops:
            
        # apply adaptive threshold
        thresh1 = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 9)

        # specify structure shape and kernel size
        # kernel size increases or decreases size of rect to be detected
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))

        # apply morphological transform
        morph = cv2.morphologyEx(thresh1, cv2.MORPH_BLACKHAT, rect_kernel)
        processed_frames.append(morph)

        # append text to file
        text = pytesseract.image_to_string(morph)
        file = open("out_text.txt", "a")
        file.write(text)
        file.close()

    return processed_frames


def crop_frame(frame):
    # crop frame to only show text
    # turn frame to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # apply threshold
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 35)

    # dilate to emphasise text blob
    crop_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 8))
    dilation = cv2.dilate(thresh, crop_kernel, iterations=2)
    bounds, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    # loop through bounds, add them to boxes to be sorted by largest then returned
    for cnt in bounds:
        x, y, w, h = cv2.boundingRect(cnt)
        region = gray[y: y+ h, x:x + w]
        boxes.append(region)

    boxes = sorted(boxes, key=lambda i: -1 * i.shape[0] * i.shape[1])

    return boxes[:2]


# construct arg parser
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, required=True, help="path to input video file")
ap.add_argument("-f", "--fps", type=int, required=False, default=10, help="desired fps of video file to be examined at")
args = vars(ap.parse_args())

# load the video file ref
video = cv2.VideoCapture(args["video"])
print("[INFO] video file loaded!")

# create new file for text output, or overwrite if exists
open("out_text.txt", "w").close()

# get desired and actual fps
desired_fps = args["fps"]
video_fps = int(video.get(cv2.CAP_PROP_FPS))

# frame number iterator
current_frame = 0

# loop over frames of the video, adjust to desired FPS
while True:
    # iterate frame count
    current_frame += 1

    # skips to next frame to adjust to desired fps
    if  current_frame % (video_fps / desired_fps) != 0:
        continue

    # grab the current frame
    frame = video.read()[1]

    # check to see if video has ended, finish
    if frame is None:
        print("[INFO] video completed, exiting...")
        break

    # store original frame
    orig_frame = frame.copy()

    # append end of frame marker to file
    file = open("out_text.txt", "a")
    file.write("\n---frame:{}---\n".format(current_frame))
    file.close()

    # detects text, returns processed frame
    processed_frame = text_detect(frame)

    # show input and output frames
    cv2.imshow("Processed Video 1", processed_frame[0])
    cv2.imshow("Processed Video 2", processed_frame[1])
    
    # prints current frame no.
    print("frame:" + str(current_frame))

    # break if q is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("[INFO] cancelled, exiting...")
        break