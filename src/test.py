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


cap = cv.VideoCapture("../shot_tests/shot9.mp4")
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

        if pixel_to_distance is None:
            pixel_to_distance, bottom_y = rim_width(rect_rim)

    if ballFound and personFound:

        # measure the amount of frames until the shot is launched
        if shotLaunched is False:
            release_count += 1

        overlap1 = get_overlap(rect_person, rect_ball)

        if len(overlap_values_person) < 4:
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

            if len(overlap_values_rim) < 2:
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

    # if key == ord('p'):
    #     cv.waitKey(-1)  # wait until any key is pressed

    if cv.waitKey(1) == 27:  # esc Key
        break

    cv.imshow("basketball", img)
    cv.setMouseCallback("basketball", draw)

cap.release()
cv.destroyAllWindows()
