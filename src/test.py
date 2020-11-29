import cv2 as cv
import numpy as np
import math
import time
import sys

(major_ver, minor_ver) = cv.__version__.split(".")[:2]

# Set up tracker.
# Instead of MIL, you can also use

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD',
                 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[7]

if int(minor_ver) < 3:
    tracker_person = cv.Tracker_create(tracker_type)
    tracker_ball = cv.Tracker_create(tracker_type)
    tracker_rim = cv.Tracker_create(tracker_type)
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
        tracker_rim = cv.TrackerCSRT_create()


# initBB_ball = None
# initBB_person = None
# circleFound = False
# personFound = False
# shotLaunched = False
#
# outputs = []


# def findPerson(outputs, img):
#     hT, wT, cT = img.shape
#     bbox = []
#     classIds = []
#     confs = []
#     for output in outputs:
#         for det in output:
#             scores = det[5:]
#             classId = np.argmax(scores)
#             confidence = scores[classId]
#             if confidence > confThreshold:
#                 w, h = int(det[2] * wT), int(det[3] * hT)
#                 x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
#                 bbox.append([x, y, w, h])
#                 classIds.append(classId)
#                 confs.append(float(confidence))
#
#     indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
#
#     for i in indices:
#         i = i[0]
#         if i < 2:
#             box = bbox[i]
#             x, y, w, h = box[0], box[1], box[2], box[3]
#             return x, y, w, h
#             # print(x,y,w,h)
#             # cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
#             # cv.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
#             #            (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
#         else:
#             return None


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
    point1 = [(box1[0] + box1[2]) / 2, box1[3]]
    point2 = [(box2[0] + box2[2]) / 2, box2[1]]
    angle_rad = math.atan2(point2[1] - point1[1], point2[0] - point1[0])
    angle = math.degrees(angle_rad)
    return angle


cap = cv.VideoCapture("../shot_tests/shot1.mp4")
overlap_values = []
boxes = []
ballFound = False
personFound = False
rimFound = False
initBB_ball = None
initBB_person = None
initBB_rim = None

count = 0
drawing = False
ix = 0
iy = 0
holder = []


def draw(event, x, y, img, flag):
    global ix, iy, drawing, holder
    # Left Mouse Button Down Pressed
    if event == 1:
        drawing = True
        ix = x
        iy = y
        holder.append([ix, iy])
    if event == 0:
        if drawing:
            ix = x
            iy = y
            holder.append([ix, iy])
    if event == 4:
        drawing = False
        holder = [holder[0], holder[len(holder) - 1]]


def rect_coordinates(arr):
    p1 = (arr[0][0], arr[0][1])
    p2 = (arr[1][0], arr[1][1])
    x_center = p1[0]
    y_center = p1[1]
    height = abs(p1[1] - p2[1])
    width = abs(p1[0] - p2[0])

    return x_center, y_center, width, height


while True:
    success, img = cap.read()
    key = cv.waitKey(0)

    if ballFound is False or personFound is False or rimFound is False:
        key
        print(holder)

        if count == 0 and len(holder) > 0:
            x, y, w, h = rect_coordinates(holder)
            initBB_ball = (x, y, w, h)
            tracker_ball.init(img, initBB_ball)
            holder = []
            ballFound = True
            count += 1

        elif count == 1 and len(holder) > 0:
            x, y, w, h = rect_coordinates(holder)
            initBB_person = (x, y, w, h)
            tracker_person.init(img, initBB_person)
            holder = []
            personFound = True
            count += 1

        elif count == 2 and len(holder) > 0:
            x, y, w, h = rect_coordinates(holder)
            initBB_rim = (x, y, w, h)
            tracker_rim.init(img, initBB_rim)
            holder = []
            rimFound = True
            count += 1

    if initBB_ball is not None:
        _, initBB_ball = tracker_ball.update(img)
        # grab the new bounding box coordinates of the object
        point1 = (int(initBB_ball[0]), int(initBB_ball[1]))
        point2 = (int(initBB_ball[0] + initBB_ball[2]), int(initBB_ball[1] + initBB_ball[3]))
        cv.rectangle(img, point1, point2, (255, 0, 0), 2, 1)
        rect_ball = [point1[0], point1[1], point2[0], point2[1]]

    if initBB_person is not None:
        _, initBB_person = tracker_person.update(img)
        # grab the new bounding box coordinates of the object
        point1 = (int(initBB_person[0]), int(initBB_person[1]))
        point2 = (int(initBB_person[0] + initBB_person[2]), int(initBB_person[1] + initBB_person[3]))
        cv.rectangle(img, point1, point2, (255, 0, 0), 2, 1)
        rect_person = [point1[0], point1[1], point2[0], point2[1]]

    if initBB_rim is not None:
        _, initBB_rim = tracker_rim.update(img)
        # grab the new bounding box coordinates of the object
        point1 = (int(initBB_rim[0]), int(initBB_rim[1]))
        point2 = (int(initBB_rim[0] + initBB_rim[2]), int(initBB_rim[1] + initBB_rim[3]))
        cv.rectangle(img, point1, point2, (255, 0, 0), 2, 1)
        rect_rim = [point1[0], point1[1], point2[0], point2[1]]

    # if key == ord('p'):
    #     cv.waitKey(-1)  # wait until any key is pressed

    if cv.waitKey(1) == 27:  # esc Key
        break

    cv.imshow("basketball", img)
    cv.setMouseCallback("basketball", draw)

cap.release()
cv.destroyAllWindows()
