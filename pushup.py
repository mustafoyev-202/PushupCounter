import cv2
import numpy as np
import cvzone
from cvzone.PoseModule import PoseDetector
import math

counter = 0
direction = 0

cap = cv2.VideoCapture(0)
wCam, hCam = 1270, 640
cap.set(3, wCam)
cap.set(4, hCam)

pd = PoseDetector(trackCon=0.70, detectionCon=0.70)


def calculate_distance(point1, point2):
    return math.hypot(point2[0] - point1[0], point2[1] - point1[1])


def angles(lmlist, p1, p2, p3, p4, p5, p6, drawpoints):
    global counter, direction

    if len(lmlist) != 0:
        point1 = lmlist[p1]
        point2 = lmlist[p2]
        point3 = lmlist[p3]
        point4 = lmlist[p4]
        point5 = lmlist[p5]
        point6 = lmlist[p6]

        if drawpoints:
            for point in [point1, point2, point3, point4, point5, point6]:
                cv2.circle(img, (point[0], point[1]), 10, (255, 0, 255), 5)
                cv2.circle(img, (point[0], point[1]), 15, (0, 255, 0), 5)

            cv2.line(img, (point1[0], point1[1]), (point2[0], point2[1]), (0, 0, 255), 6)
            cv2.line(img, (point2[0], point2[1]), (point3[0], point3[1]), (0, 0, 255), 6)
            cv2.line(img, (point4[0], point4[1]), (point5[0], point5[1]), (0, 0, 255), 6)
            cv2.line(img, (point5[0], point5[1]), (point6[0], point6[1]), (0, 0, 255), 6)
            cv2.line(img, (point1[0], point1[1]), (point4[0], point4[1]), (0, 0, 255), 6)

        left_hand_angle = math.degrees(math.atan2(point3[1] - point2[1], point3[0] - point2[0]) -
                                       math.atan2(point1[1] - point2[1], point1[0] - point2[0]))
        right_hand_angle = math.degrees(math.atan2(point6[1] - point5[1], point6[0] - point5[0]) -
                                        math.atan2(point4[1] - point5[1], point4[0] - point5[0]))

        # Calculate the scale factor based on shoulder width (distance between p2 and p4)
        shoulder_width = calculate_distance(point2, point4)
        scale_factor = shoulder_width / 100  # 100 is an arbitrary base value

        left_hand_angle_norm = int(np.interp(left_hand_angle, [-30, 180], [100, 0]) * scale_factor)
        right_hand_angle_norm = int(np.interp(right_hand_angle, [34, 173], [100, 0]) * scale_factor)

        left, right = left_hand_angle_norm, right_hand_angle_norm

        if left >= 70 and right >= 70:
            if direction == 0:
                counter += 0.5
                direction = 1
        if left <= 70 and right <= 70:
            if direction == 1:
                counter += 0.5
                direction = 0

        cv2.rectangle(img, (0, 0), (120, 120), (255, 0, 0), -1)
        cv2.putText(img, str(int(counter)), (20, 70), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.6, (0, 0, 255), 7)

        leftval = np.interp(left, [0, 100], [400, 200])
        rightval = np.interp(right, [0, 100], [400, 200])

        cv2.putText(img, 'R', (24, 195), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 7)
        cv2.rectangle(img, (8, 200), (50, 400), (0, 255, 0), 5)
        cv2.rectangle(img, (8, int(rightval)), (50, 400), (255, 0, 0), -1)

        cv2.putText(img, 'L', (962, 195), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 7)
        cv2.rectangle(img, (952, 200), (995, 400), (0, 255, 0), 5)
        cv2.rectangle(img, (952, int(leftval)), (995, 400), (255, 0, 0), -1)

        if left > 70:
            cv2.rectangle(img, (952, int(leftval)), (995, 400), (0, 0, 255), -1)

        if right > 70:
            cv2.rectangle(img, (8, int(leftval)), (50, 400), (0, 0, 255), -1)


while True:
    ret, img = cap.read()
    if not ret:
        cap = cv2.VideoCapture(0)
        continue

    img = cv2.resize(img, (1000, 500))
    cvzone.putTextRect(img, 'AI Push Up Counter', [345, 30], thickness=2, border=2, scale=2.5)
    pd.findPose(img, draw=0)
    lmlist, bbox = pd.findPosition(img, draw=0, bboxWithHands=0)

    angles(lmlist, 11, 13, 15, 12, 14, 16, drawpoints=1)

    cv2.imshow('frame', img)
    cv2.waitKey(1)
