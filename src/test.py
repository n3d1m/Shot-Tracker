import imutils
from imutils.video import VideoStream
from imutils.video import FPS
from shapely.geometry import Polygon
import argparse
import time
import math
import cv2 as cv
import numpy as np

(major_ver, minor_ver) = cv.__version__.split(".")[:2]

# Set up tracker.
# Instead of MIL, you can also use

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD',
                 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[7]

if int(minor_ver) < 3:
    tracker_person = cv.Tracker_create(tracker_type)
    tracker_ball = cv.Tracker_create(tracker_type)
else:
    if tracker_type == 'BOOSTING':
        tracker_person = cv.TrackerBoosting_create()
        tracker_ball = cv.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker_person = cv.TrackerMIL_create()
        tracker_ball = cv.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker_person = cv.TrackerKCF_create()
        tracker_ball = cv.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker_person = cv.TrackerTLD_create()
        tracker_ball = cv.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker_person = cv.TrackerMedianFlow_create()
        tracker_ball = cv.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker_person = cv.TrackerGOTURN_create()
        tracker_ball = cv.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker_person = cv.TrackerMOSSE_create()
        tracker_ball = cv.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker_person = cv.TrackerCSRT_create()
        tracker_ball = cv.TrackerCSRT_create()

whT = 320
confThreshold = 0.5
nmsThreshold = 0.2
classesFile = "../yolo/coco.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

## Model Files
modelConfiguration = "../yolo/darknet/cfg/yolov3.cfg"
modelWeights = "../yolo/darknet/yolov3.weights"
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

initBB_ball = None
initBB_person = None
circleFound = False
personFound = False


def findPerson(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    for i in indices:
        i = i[0]
        if i < 2:
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            return x, y, w, h
            # print(x,y,w,h)
            # cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            # cv.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
            #            (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        else:
            return None


def findBall(gray_frame, rows):
    circles = cv.HoughCircles(gray_frame, cv.HOUGH_GRADIENT, 1, rows / 8,
                              param1=100, param2=30,
                              minRadius=15, maxRadius=25)

    if circles is not None:
        circles = np.array(np.uint16(np.around(circles)))
        for i in circles[0, :]:
            radius = i[2]
            x, y, w, h = i[0] - radius, i[1] - radius, radius * 2.25, radius * 2.25

            return x, y, w, h

    else:
        return None


def get_overlap(box_1, box_2):
    # area_1 = float((box_1[2] - box_1[0]) * box_1[3] - box_1[1])
    # area_2 = float((box_2[2] - box_2[0]) * box_2[3] - box_2[1])

    width = calculate_intersection(box_1[0], box_1[2], box_2[0], box_2[2])
    height = calculate_intersection(box_1[1], box_1[3], box_2[1], box_2[3])

    area_delta = width * height

    return area_delta


def calculate_intersection(a0, a1, b0, b1):
    if a0 >= b0 and a1 <= b1:  # Contained
        intersection = a1 - a0
    elif a0 < b0 and a1 > b1:  # Contains
        intersection = b1 - b0
    elif a0 < b0 < a1:  # Intersects right
        intersection = a1 - b0
    elif a1 > b1 > a0:  # Intersects left
        intersection = b1 - a0
    else:  # No intersection (either side)
        intersection = 0

    return intersection


def calculate_angle(box1, box2):
    point1 = [(box1[0] + box1[2])/2, box1[3]]
    point2 = [(box2[0] + box2[2]) / 2, box2[1]]
    angle_rad = math.atan2(point2[1] - point1[1], point2[0] - point1[0])
    angle = math.degrees(angle_rad)
    print(angle)


cap = cv.VideoCapture("test.mp4")
overlap_values = []

while True:
    success, img = cap.read()
    key = cv.waitKey(1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)
    rows = gray.shape[0]

    if key == ord('p'):
        cv.waitKey(-1)  # wait until any key is pressed

    if cv.waitKey(1) == 27:  # esc Key
        break

    # person detection block
    if personFound is False:
        blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputNames)
        BB_person = findPerson(outputs, img)

        if BB_person is not None:
            x_p, y_p, w_p, h_p = BB_person
            initBB_person = (x_p, y_p, w_p, h_p * 1.25)
            tracker_person.init(img, initBB_person)
            personFound = True

    if initBB_person is not None:
        _, initBB_person = tracker_person.update(img)
        # grab the new bounding box coordinates of the object
        p1 = (int(initBB_person[0]), int(initBB_person[1]))
        p2 = (int(initBB_person[0] + initBB_person[2]), int(initBB_person[1] + initBB_person[3]))
        cv.rectangle(img, p1, p2, (255, 0, 0), 2, 1)
        rect_person = [p1[0], p1[1], p2[0], p2[1]]

    # ball detection block
    if circleFound is False:
        BB_ball = findBall(gray, rows)

        if BB_ball is not None:
            x_b, y_b, w_b, h_b = BB_ball
            initBB_ball = (x_b, y_b, w_b, h_b)
            tracker_ball.init(img, initBB_ball)
            circleFound = True

    if initBB_ball is not None:
        _, initBB_ball = tracker_ball.update(img)
        # grab the new bounding box coordinates of the object
        p1 = (int(initBB_ball[0]), int(initBB_ball[1]))
        p2 = (int(initBB_ball[0] + initBB_ball[2]), int(initBB_ball[1] + initBB_ball[3]))
        cv.rectangle(img, p1, p2, (255, 0, 0), 2, 1)
        rect_ball = [p1[0], p1[1], p2[0], p2[1]]

    cv.imshow('Image', img)

    # This is where most of the analysis is done
    if circleFound and personFound:

        overlap = get_overlap(rect_person, rect_ball)

        if len(overlap_values) < 3:
            overlap_values.append(overlap)
        else:
            overlap_values = overlap_values[1:]
            overlap_values.append(overlap)

        if sum(overlap_values) == 0:
            calculate_angle(rect_ball, rect_person)
            cv.waitKey(-1)

        # box1 = rect_ball
        # box2 = rect_person
        # overlap_bool = overlap(box1, box2)
        # print(overlap_percentage)

cap.release()
cv.destroyAllWindows()
