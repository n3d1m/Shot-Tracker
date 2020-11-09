import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import time
import cv2 as cv
import numpy as np

(major_ver, minor_ver) = cv.__version__.split(".")[:2]


# Set up tracker.
# Instead of MIL, you can also use

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD',
                 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[7]

if int(minor_ver) < 3:
    tracker = cv.Tracker_create(tracker_type)
else:
    if tracker_type == 'BOOSTING':
        tracker = cv.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv.TrackerCSRT_create()

initBB = None
circleFound = False
cap = cv.VideoCapture("test.mp4")

while True:
    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    gray = cv.medianBlur(gray, 5)
    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                              param1=100, param2=30,
                              minRadius=15, maxRadius=25)

    if circles is not None and circleFound is False:
        circles = np.array(np.uint16(np.around(circles)))
        # circles = np.append(circles, 1)
        print(circles)
        # initBB = tuple(circles)
        # tracker.init(frame, initBB)
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(frame, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            # circle = cv.circle(frame, center, radius, (255, 0, 255), 3)
            initBB = (i[0]-radius, i[1]-radius, radius*2.25, radius*2.25)
            tracker.init(frame, initBB)
            circleFound = True
            # cv.imshow("video", frame)

    if initBB is not None:
        _, initBB = tracker.update(frame)
        # grab the new bounding box coordinates of the object
        p1 = (int(initBB[0]), int(initBB[1]))
        p2 = (int(initBB[0] + initBB[2]), int(initBB[1] + initBB[3]))
        cv.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

    # else:
    cv.imshow('tracking', frame)

    if cv.waitKey(1) == 27:  # esc Key
        break

cap.release()
cv.destroyAllWindows()
