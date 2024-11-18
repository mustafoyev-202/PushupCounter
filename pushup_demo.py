import cv2
import numpy as np
import cvzone
from cvzone.PoseModule import PoseDetector
import math
import argparse


def calculate_distance(point1, point2):
    return math.hypot(point2[0] - point1[0], point2[1] - point1[1])


def process_video(input_path, output_path):
    # Open input video
    cap = cv2.VideoCapture(input_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize tracking variables
    counter = 0
    direction = 0
    pd = PoseDetector(trackCon=0.70, detectionCon=0.70)

    def angles(img, lmlist, p1, p2, p3, p4, p5, p6, drawpoints):
        nonlocal counter, direction

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

            # Calculate the scale factor based on shoulder width
            shoulder_width = calculate_distance(point2, point4)
            scale_factor = shoulder_width / 100

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

            cv2.putText(img, 'L', (width - 48, 195), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 7)
            cv2.rectangle(img, (width - 50, 200), (width - 8, 400), (0, 255, 0), 5)
            cv2.rectangle(img, (width - 50, int(leftval)), (width - 8, 400), (255, 0, 0), -1)

            if left > 70:
                cv2.rectangle(img, (width - 50, int(leftval)), (width - 8, 400), (0, 0, 255), -1)

            if right > 70:
                cv2.rectangle(img, (8, int(rightval)), (50, 400), (0, 0, 255), -1)

        return img

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Add title to the frame
        cvzone.putTextRect(frame, 'AI Push Up Counter', [width // 2 - 150, 30], thickness=2, border=2, scale=2.5)

        # Detect pose
        pd.findPose(frame, draw=0)
        lmlist, bbox = pd.findPosition(frame, draw=0, bboxWithHands=0)

        # Process angles and draw
        frame = angles(frame, lmlist, 11, 13, 15, 12, 14, 16, drawpoints=1)

        # Write the frame
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Total push-ups counted: {int(counter)}")
    print(f"Output video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Push-Up Counter Video Processor')
    parser.add_argument('input', help='Input video file path')
    parser.add_argument('-o', '--output', help='Output video file path',
                        default='pushup_counter_output.mp4')

    args = parser.parse_args()
    process_video(args.input, args.output)


if __name__ == '__main__':
    main()
