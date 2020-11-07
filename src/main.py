from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='test.mp4')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()

if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:

    backSub = cv.createBackgroundSubtractorKNN()
capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))

if not capture.isOpened:
    print('Unable to open: ' + args.input)
    exit(0)

n = 0
while True:
    ret, frame = capture.read()
    if frame is None:
        break

    fgMask = backSub.apply(frame)

    cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    detected_circles = cv.HoughCircles(fgMask,
                                       cv.HOUGH_GRADIENT, 1, 20, param1=50,
                                       param2=30, minRadius=1, maxRadius=30)

    # Draw circles that are detected.
    if detected_circles is not None:

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]

            # Draw the circumference of the circle.
            cv.circle(fgMask, (a, b), r, (0, 255, 0), 2)

            # Draw a small circle (of radius 1) to show the center.
            cv.circle(fgMask, (a, b), 1, (0, 0, 255), 3)
            cv.imshow("Detected Circle", fgMask)
            cv.waitKey(0)

    # cv.imshow('Frame', frame)
    # cv.imshow('FG Mask', fgMask)

    # cv.imwrite("frames/frame-{:03d}.jpg".format(n), fgMask)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
    n += 1
