import argparse
import numpy as np
import cv2
import pytesseract

def text_detect(frame):
    # takes frames, finds text and appended to txt
    # turn frame to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, -10)

    processed_frame = cv2.GaussianBlur(thresh,(3,3),0)

    # append text to file
    text = pytesseract.image_to_string(processed_frame, config='--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    file = open("out_text.txt", "a")
    file.write(text)
    file.close()

    return processed_frame

# draw_rectangle global_vars
drawing = False
ix, iy = -1, -1
coords = []
frame = ""
clone = ""
roi_regions = []

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, frame, clone, coords, roi_regions
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        coords = [(x, y)]
        roi_regions.append(coords)
        clone = frame.copy()
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            frame = clone.copy()
            cv2.rectangle(frame, (ix, iy), (x, y), (0, 255, 0), 2)
            a = x
            b = y
            if a != x | b != y:
                cv2.rectangle(frame, (ix, iy), (x, y), (0, 0, 0), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        coords.append((x, y))
        frame = clone.copy()
        cv2.rectangle(frame, (ix, iy), (x, y), (0, 255, 0), 1)


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

# select ROI's
frame = video.read()[1]
frame = np.array(frame)
cv2.namedWindow('ROIs', cv2.WINDOW_GUI_NORMAL)
cv2.setMouseCallback('ROIs', draw_rectangle)
cv2.setWindowProperty('ROIs', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    cv2.imshow('ROIs', frame)

    # undo last rectangle
    key = cv2.waitKey(1) & 0xFF
    if key == 8:
        del roi_regions[-1]
        del coords[-1]
        frame = clone.copy()

    # break when finished/ enter pressed
    if key == 13:
        cv2.destroyAllWindows()
        break

# loop over cropped frames of the video, adjust to desired FPS
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

    # append end of frame marker to file
    file = open("out_text.txt", "a")
    file.write("\n---frame:{}---\n".format(current_frame))
    file.close()

    # crop frame to ROIs only, loop and process
    n = 0
    for roi in roi_regions:
        x1, y1 = roi[0]
        x2, y2 = roi[1]
        # make x1, y1 top left coords
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        crop = frame[y1:y2, x1:x2]

        # detects text, returns processed frame
        processed_frame = text_detect(crop)

        # individual window name to allow display of multiple
        n += 1
        name = "ROI: {}".format(n)

        # show input and output frames
        cv2.imshow(name, processed_frame)
    
    # prints current frame no.
    print("frame:" + str(current_frame))

    # break if q is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("[INFO] cancelled, exiting...")
        break