import argparse
import argparse
import numpy as np
import cv2
import pytesseract

def text_detect(frame):
    # takes frames, finds text and appended to txt

    # turn frame to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # perform OTSU threshold
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # specify structure shape and kernel size
    # kernel size increases or decreases size of rect to be detected
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

    # apply dilation on threshold image
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)
    
    # find contours
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # copy input frame
    processed_frame = frame.copy()

    # loop through contours that could be text
    # crop those rects and pass to pytesseract for extraction
    # append extracted text to text file
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # draw rectangle on copied image
        cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # crop text block to give input for OCR
        cropped = processed_frame[y:y + h, x:x +w]

        # apply OCR to cropped areas
        text = pytesseract.image_to_string(cropped)

        # append recognised text to file
        file = open("out_text.txt", "a")
        file.write(text + "\n")
        file.close()

    return processed_frame


# construct arg parser
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, required=True, help="path to input video file")
ap.add_argument("-f", "--fps", type=int, required=False, default=10, help="desired fps of video file to be examined at")
args = vars(ap.parse_args())

# load the video file ref
video = cv2.VideoCapture(args["video"])
print("[INFO] video file loaded!")

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
    file = open("out_text.txt", "a+")
    file.write("\n---frame:{}---\n".format(current_frame))
    file.close()

    # detects text, returns processed frame
    processed_frame = text_detect(frame)

    # show input and output frames
    cv2.imshow("Processed Video", processed_frame)
    
    # prints current frame no.
    print("frame:" + str(current_frame))

    # break if q is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("[INFO] cancelled, exiting...")
        break