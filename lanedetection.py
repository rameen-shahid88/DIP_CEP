import cv2
import numpy as np
import os
from collections import deque

# Global variables
FRAME_W, FRAME_H = 960, 540
history = deque(maxlen=12)
prev_left = None
prev_right = None

# Edge Detection
def get_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)   # increase contrast
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 70, 180)
    return edges


# ROI
def region_of_interest(edges):
    h, w = edges.shape
    mask = np.zeros_like(edges)

    polygon = np.array([[
        (int(0.05*w), h),
        (int(0.95*w), h),
        (int(0.65*w), int(0.55*h)),
        (int(0.35*w), int(0.55*h))
    ]])

    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(edges, mask)


# HOUGH + FILTER
def get_lane_lines(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50,
                            minLineLength=50, maxLineGap=100)

    left, right = [], []

    if lines is None:
        return None, None

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x2 == x1:
            continue

        slope = (y2 - y1) / (x2 - x1)

        # filter noise
        if abs(slope) < 0.5 or abs(slope) > 2:
            continue

        if slope < 0:
            left.append((x1, y1, x2, y2))
        else:
            right.append((x1, y1, x2, y2))

    return left, right


# Average lines
def average_line(lines):
    if not lines:
        return None

    x, y = [], []

    for x1, y1, x2, y2 in lines:
        x += [x1, x2]
        y += [y1, y2]

    poly = np.polyfit(y, x, 1)

    y1 = FRAME_H
    y2 = int(FRAME_H * 0.6)

    x1 = int(np.polyval(poly, y1))
    x2 = int(np.polyval(poly, y2))

    return (x1, y1, x2, y2)


# Direction (stable)
def get_direction(frame, left_line, right_line):
    h, w, _ = frame.shape
    center = w // 2

    lane_width = int(w * 0.4)  # estimated lane width

    if left_line is not None and right_line is not None:
        lane_center = (left_line[0] + right_line[0]) // 2


    elif left_line is not None:
        lane_center = left_line[0] + int(lane_width * 0.9)


    elif right_line is not None:

        lane_center = right_line[0] - int(lane_width * 0.9)

    else:
        return "STRAIGHT"

    deviation = lane_center - center

    # Dead zone
    if abs(deviation) < w * 0.08:
        return "STRAIGHT"
    elif deviation > 0:
        return "RIGHT"
    else:
        return "LEFT"


# Draw
def draw(frame, left_line, right_line, direction):
    overlay = frame.copy()

    if left_line and right_line:
        pts = np.array([
            [left_line[0], left_line[1]],
            [left_line[2], left_line[3]],
            [right_line[2], right_line[3]],
            [right_line[0], right_line[1]]
        ])
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)

    if left_line:
        cv2.line(frame, (left_line[0], left_line[1]),
                 (left_line[2], left_line[3]), (0,255,0), 5)

    if right_line:
        cv2.line(frame, (right_line[0], right_line[1]),
                 (right_line[2], right_line[3]), (0,255,0), 5)

    cv2.putText(frame, f"Direction: {direction}",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), 2)

    return frame


# Main
def process_frame(frame):
    edges = get_edges(frame)
    roi = region_of_interest(edges)

    left, right = get_lane_lines(roi)

    global prev_left, prev_right

    left_line = average_line(left)
    right_line = average_line(right)

    # Smooth lines over time
    alpha = 0.7

    if left_line is not None:
        if prev_left is not None:
            left_line = tuple(
                int(alpha * pl + (1 - alpha) * cl)
                for pl, cl in zip(prev_left, left_line)
            )
        prev_left = left_line

    if right_line is not None:
        if prev_right is not None:
            right_line = tuple(
                int(alpha * pl + (1 - alpha) * cl)
                for pl, cl in zip(prev_right, right_line)
            )
        prev_right = right_line

    direction = get_direction(frame, left_line, right_line)

    # smoothing
    history.append(direction)
    direction = max(set(history), key=history.count)

    output = draw(frame.copy(), left_line, right_line, direction)

    return output


# Video loop
def run(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))

        output = process_frame(frame)
        cv2.imshow("Lane Detection", output)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# Main
folder = r"G:\Computer Engineering\6th Semester\DIP\Project\DIP Project Videos"

for file in os.listdir(folder):
    if file.endswith(('.mp4', '.avi', '.mov')):
        print("Processing:", file)
        run(os.path.join(folder, file))