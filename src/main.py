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

whT = 320
confThreshold = 0.5
nmsThreshold = 0.2
classesFile = "../yolo/darknet/data/basketball.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

## Model Files
modelConfiguration = "../yolo/darknet/cfg/basketball.cfg"
modelWeights = "../yolo/darknet/best.weights"
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


def find_objects(img):
    hT, wT, cT = img.shape
    blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
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
        label = classNames[classIds[i]]
        print(label)
        if label == 'ball' or label == 'basket' or label == 'walk' or label == 'shoot' or label == 'basket':
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]

            return [x, y, w, h, label]

        else:

            return None


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


def get_overlap(box_1, box_2):
    width = calculate_intersection(box_1[0], box_1[2], box_2[0], box_2[2])
    height = calculate_intersection(box_1[1], box_1[3], box_2[1], box_2[3])

    area_delta = width * height

    return area_delta


def calculate_angle(box1, box2):
    point1 = [box1[0], box1[1]]
    point2 = [(box2[0] + box2[2]) / 2, box2[1]]
    angle_rad = math.atan2(point2[1] - point1[1], point2[0] - point1[0])
    angle_val = math.degrees(angle_rad)
    return angle_val


def calculate_shot_values(f_count, fps, height_arr, h_l, conversion, bottom, r_count):
    distance = 4.572  # distance of free throw in meters
    gravity = 9.81
    radius = 0.23  # radius of the rim
    r_half = radius / 2
    height = 3.048  # height of 10 foot net in m

    time_val = f_count / fps
    release_time = r_count / fps
    h_max = min(height_arr)
    h_delta = abs(h_max - h_l) * conversion
    h_peak = abs(h_max - bottom) * conversion

    print(h_l * conversion)

    calc_min = (r_half / (2 * radius)) * math.pow(1 - (r_half / radius), -0.5) + 2 * (
            (height - h_l * conversion) / distance)
    theta_min = math.degrees(math.atan(calc_min))

    v_x = distance / time_val
    v_y = math.sqrt(2 * gravity * h_delta)
    v_delta = math.sqrt(v_x ** 2 + v_y ** 2)
    release_angle = math.degrees(math.atan2(v_y, v_x))

    calc_actual = (gravity*time_val - v_delta*math.degrees(math.sin(theta_min))) / (v_delta*math.degrees(math.cos(theta_min)))
    theta_actual = math.degrees(math.atan(calc_actual))

    if theta_actual < 0:
        theta_actual = 90 - abs(theta_actual)

    return {
        'V_x': v_x,
        'V_y': v_y,
        'V': v_delta,
        'Release Angle': release_angle,
        'Peak': h_peak,
        'Release Time': release_time,
        'Minimum Angle of Entry': theta_min,
        'Angle of Entry': theta_actual
    }


def rim_width(box):
    width = 0.46
    height = 3.048
    x_1 = box[0]
    x_2 = box[2]
    distance = abs(x_1 - x_2) * 0.8
    conversion = width / distance
    rim_height = height / conversion
    y_bottom = round(box[1] + rim_height)

    return conversion, y_bottom


def velocity_calc(arr, fps, pix_convert):
    starting_point = arr[0]
    end_point = arr[1]
    delta_time = 1 / fps

    delta_x = abs(starting_point[0] - end_point[0])
    delta_y = abs(starting_point[1] - end_point[1]) * pixel_to_distance
    v_x = delta_x / delta_time
    v_y = delta_y / delta_time

    print(v_x, pix_convert)


cap = cv.VideoCapture('../shot_tests/shot4.mp4')
overlap_values_person = []
overlap_values_rim = []
outputs = []
FPS = None
ballFound = False
personFound = False
rimFound = False
initBB_ball = None
initBB_person = None
initBB_rim = None
shotLaunched = False
pixel_to_distance = None
bottom_y = None
h_launch = None
frame_count = 0
release_count = 0
ball_positions = []
hit_rim = False

while True:
    success, img = cap.read()
    # start_time = time.time()

    if ballFound is False or personFound is False or rimFound is False:

        object_params = find_objects(img)

        if object_params is not None:
            object_label = object_params[4]

            if object_label == 'ball' and ballFound is False:
                initBB_ball = (object_params[0], object_params[1], object_params[2], object_params[3])
                tracker_ball.init(img, initBB_ball)
                ballFound = True
            if (object_label == 'shoot' or object_label == 'walk') and personFound is False:
                initBB_person = (object_params[0], object_params[1] - 100, object_params[2], object_params[3] * 1.9)
                tracker_person.init(img, initBB_person)
                personFound = True
            if object_label == 'basket' and rimFound is False:
                initBB_rim = (object_params[0], object_params[1], object_params[2], object_params[3])
                tracker_rim.init(img, initBB_rim)
                rimFound = True

    if initBB_ball is not None:
        _, initBB_ball = tracker_ball.update(img)
        # grab the new bounding box coordinates of the object
        p1 = (int(initBB_ball[0]), int(initBB_ball[1]))
        p2 = (int(initBB_ball[0] + initBB_ball[2]), int(initBB_ball[1] + initBB_ball[3]))
        cv.rectangle(img, p1, p2, (255, 0, 0), 2, 1)
        rect_ball = [p1[0], p1[1], p2[0], p2[1]]

    if initBB_person is not None:
        _, initBB_person = tracker_person.update(img)
        # grab the new bounding box coordinates of the object
        p1 = (int(initBB_person[0]), int(initBB_person[1]))
        p2 = (int(initBB_person[0] + initBB_person[2]), int(initBB_person[1] + initBB_person[3]))
        cv.rectangle(img, p1, p2, (255, 0, 0), 2, 1)
        rect_person = [p1[0], p1[1], p2[0], p2[1]]

    if initBB_rim is not None:
        _, initBB_rim = tracker_rim.update(img)
        # grab the new bounding box coordinates of the object
        p1 = (int(initBB_rim[0]), int(initBB_rim[1]))
        p2 = (int(initBB_rim[0] + initBB_rim[2]), int(initBB_rim[1] + initBB_rim[3]))
        cv.rectangle(img, p1, p2, (255, 0, 0), 2, 1)
        rect_rim = [p1[0], p1[1], p2[0], p2[1]]

        # pixel conversion based on rim width happens here
        if pixel_to_distance is None:
            pixel_to_distance, bottom_y = rim_width(rect_rim)

    # This is where most of the analysis is done
    if ballFound and personFound:

        # measure the amount of frames until the shot is launched
        if shotLaunched is False:
            release_count += 1

        overlap1 = get_overlap(rect_person, rect_ball)

        if len(overlap_values_person) < 1:
            overlap_values_person.append(overlap1)
        else:
            overlap_values_person = overlap_values_person[1:]
            overlap_values_person.append(overlap1)

        person_bottom = rect_person[3]
        ball_top = rect_ball[1]

        if sum(overlap_values_person) == 0 and shotLaunched is False and person_bottom > ball_top:
            # angle = calculate_angle(rect_ball, rect_person)
            # outputs.append(angle)
            shotLaunched = True
            h_launch = bottom_y - rect_ball[3]
            print('SHOT LAUNCHED', bottom_y, rect_ball)

        if shotLaunched:
            if hit_rim is False:
                frame_count += 1
                ball_positions.append(rect_ball[3])

            overlap2 = get_overlap(rect_rim, rect_ball)

            if len(overlap_values_rim) < 3:
                overlap_values_rim.append(overlap2)
            else:
                overlap_values_rim = overlap_values_rim[1:]
                overlap_values_rim.append(overlap2)

            if sum(overlap_values_rim) != 0:
                FPS = cap.get(cv.CAP_PROP_FPS)
                # print(ball_positions, frame_count, FPS)
                hit_rim = True
                parameters = calculate_shot_values(frame_count, FPS, ball_positions,
                                                   h_launch, pixel_to_distance, bottom_y, release_count)
                print(parameters)
                cv.waitKey(-1)

                if overlap_values_rim[2] > 1500:
                    print('SHOT MADE')

            # if sum(overlap_values_rim)

            # if frame_count == 2:
            #     velocity_calc(ball_positions, FPS, pixel_to_distance)
            #     cv.waitKey(-1)

    cv.imshow("basketball", img)
    # cv.waitKey(0)
    if cv.waitKey(1) == 27:  # esc Key
        break

cap.release()
cv.destroyAllWindows()
