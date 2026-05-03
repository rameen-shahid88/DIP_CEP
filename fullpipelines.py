import cv2
import numpy as np
import os
import glob
from collections import deque
from ultralytics import YOLO

MAX_DISPLAY_W   = 800
MAX_DISPLAY_H   = 600
MIN_DISPLAY_W   = 400
MIN_DISPLAY_H   = 300

CONF_THRESH     = 0.4
HISTORY_LEN     = 12
SMOOTHING_ALPHA = 0.7
YOLO_SKIP_FRAMES = 3

#classes for yolo
IMPORTANT_CLASSES = {
    'car', 'person', 'bicycle', 'motorcycle', 'bus', 'truck',
    'bench', 'chair', 'potted plant', 'cat', 'dog',
    'stop sign', 'traffic light'
}

FOLDER_PATH = r"C:\Users\Rameen Shahid\Downloads\DIP Project Videos"

C_GREEN  = (0,  220,  80)
C_RED    = (30,  30, 230)
C_YELLOW = (0,  210, 255)
C_WHITE  = (240, 240, 240)
C_DARK   = (18,  18,  18)
C_ACCENT = (0,  180, 255)

class LaneDetector:
    def __init__(self):
        self.prev_left  = None   # last frame's left lane line
        self.prev_right = None   # last frame's right lane line
        self.history    = deque(maxlen=HISTORY_LEN)   # keeps past direction decisions

    def get_edges(self, frame):
        # Turn the image into edges (white lines on black background)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)          # makes edges pop out
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.Canny(blur, 70, 180)        # Canny edge detection

    def region_of_interest(self, edges):
        # Only look at the bottom part of the image where lanes usually are
        h, w = edges.shape
        mask = np.zeros_like(edges)
        # Polygon that covers the road area
        poly = np.array([[
            (int(0.05 * w), h),
            (int(0.95 * w), h),
            (int(0.65 * w), int(0.55 * h)),
            (int(0.35 * w), int(0.55 * h))
        ]])
        cv2.fillPoly(mask, poly, 255)
        return cv2.bitwise_and(edges, mask)

    def get_raw_lines(self, edges):
        # Find all straight line segments from the edge image
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50,
                                minLineLength=50, maxLineGap=100)
        left, right = [], []
        if lines is None:
            return left, right
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            # Ignore lines that are too flat or too steep
            if abs(slope) < 0.5 or abs(slope) > 2:
                continue
            if slope < 0:
                left.append((x1, y1, x2, y2))   # left lane has negative slope
            else:
                right.append((x1, y1, x2, y2))  # right lane has positive slope
        return left, right

    def average_line(self, lines):
        # Combine many short line segments into one long line
        if not lines:
            return None
        xs, ys = [], []
        for x1, y1, x2, y2 in lines:
            xs += [x1, x2]
            ys += [y1, y2]
        # Fit a line through all points
        poly = np.polyfit(ys, xs, 1)
        y_bot = FRAME_H
        y_top = int(FRAME_H * 0.6)
        return (int(np.polyval(poly, y_bot)), y_bot,
                int(np.polyval(poly, y_top)), y_top)

    def _smooth(self, new_line, prev_line):
        # Smooth lane lines so they don't shake too much
        if new_line is None:
            return prev_line
        if prev_line is None:
            return new_line
        # Blend old and new line
        return tuple(
            int(SMOOTHING_ALPHA * p + (1 - SMOOTHING_ALPHA) * n)
            for p, n in zip(prev_line, new_line)
        )

    def get_direction(self, left_line, right_line):
        # Decide if the car should go straight, left, or right based on lane lines
        w = FRAME_W
        center = w // 2
        lane_w = int(w * 0.4)

        if left_line and right_line:
            lane_center = (left_line[0] + right_line[0]) // 2
        elif left_line:
            lane_center = left_line[0] + int(lane_w * 0.9)
        elif right_line:
            lane_center = right_line[0] - int(lane_w * 0.9)
        else:
            return "STRAIGHT"

        dev = lane_center - center
        if abs(dev) < w * 0.08:
            return "STRAIGHT"
        return "RIGHT" if dev > 0 else "LEFT"

    def process(self, frame):
        # Main lane detection pipeline for one frame
        edges = self.get_edges(frame)
        roi = self.region_of_interest(edges)
        left_raw, right_raw = self.get_raw_lines(roi)

        left_line = self._smooth(self.average_line(left_raw), self.prev_left)
        right_line = self._smooth(self.average_line(right_raw), self.prev_right)

        # Remember current lines for next frame
        if left_line:
            self.prev_left = left_line
        if right_line:
            self.prev_right = right_line

        direction = self.get_direction(left_line, right_line)
        self.history.append(direction)
        # Use the most common direction from recent history
        direction = max(set(self.history), key=self.history.count)

        return left_line, right_line, direction


def draw_lanes(frame, left_line, right_line):
    # Draw the lane area and lane lines on the frame
    overlay = frame.copy()
    if left_line and right_line:
        # Fill the polygon between both lane lines
        pts = np.array([
            [left_line[0], left_line[1]],
            [left_line[2], left_line[3]],
            [right_line[2], right_line[3]],
            [right_line[0], right_line[1]]
        ], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], C_GREEN)
        frame = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)
    # Draw the two lane lines themselves
    for line, colour in [(left_line, C_GREEN), (right_line, C_GREEN)]:
        if line:
            cv2.line(frame, (line[0], line[1]), (line[2], line[3]),
                     colour, 4, cv2.LINE_AA)
    return frame


def draw_detections(frame, boxes, names):
    # Draw bounding boxes and labels for detected objects
    h, w = frame.shape[:2]
    for box in boxes:
        name = names[int(box.cls[0])]
        if name not in IMPORTANT_CLASSES:
            continue  # ignore uninteresting objects
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        bh = y2 - y1
        cx = (x1 + x2) // 2
        # Mark as threat if the object is big and in the middle of the frame
        is_threat = (bh > h * 0.25 and w * 0.25 <= cx <= w * 0.75)
        colour = C_RED if is_threat else C_ACCENT

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        label = f"{name}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
        pad = 4
        # Draw background for text
        cv2.rectangle(frame, (x1, y1 - th - 2 * pad), (x1 + tw + 2 * pad, y1), colour, -1)
        cv2.putText(frame, label, (x1 + pad, y1 - pad),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, C_DARK, 1, cv2.LINE_AA)
    return frame


def draw_hud(frame, lane_direction, avoid_direction, obj_count):
    # Draw the bottom status bar with lane, avoid, and object info
    h, w = frame.shape[:2]

    # Height of the bottom bar
    bar_h = max(40, min(90, int(h * 0.10)))
    y0 = h - bar_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), C_DARK, -1)
    frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
    cv2.line(frame, (0, y0), (w, y0), C_ACCENT, 2)

    # Pick font sizes based on window width
    scale = w / 800
    font_small = max(0.35, min(0.6, 0.4 * scale))
    font_large = max(0.55, min(1.0, 0.7 * scale))

    # Split bar into three equal columns
    sec_w = w // 3
    centers = [sec_w // 2, sec_w + sec_w // 2, 2 * sec_w + sec_w // 2]

    def put(center_x, label, value, val_color=C_WHITE):
        label_y = y0 + int(bar_h * 0.35)
        value_y = y0 + int(bar_h * 0.75)
        (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, font_small, 1)
        (vw, _), _ = cv2.getTextSize(value, cv2.FONT_HERSHEY_DUPLEX, font_large, 2)
        cv2.putText(frame, label,
                    (center_x - lw // 2, label_y),
                    cv2.FONT_HERSHEY_DUPLEX, font_small,
                    C_ACCENT, 1, cv2.LINE_AA)
        cv2.putText(frame, value,
                    (center_x - vw // 2, value_y),
                    cv2.FONT_HERSHEY_DUPLEX, font_large,
                    val_color, 2, cv2.LINE_AA)

    # Column 1: lane direction
    lane_col = C_GREEN if lane_direction == "STRAIGHT" else C_YELLOW
    put(centers[0], "LANE", lane_direction, lane_col)

    # Column 2: avoid direction (which way to steer clear of obstacles)
    avoid_col = C_GREEN if avoid_direction == "GO" else C_YELLOW
    put(centers[1], "TURN (Object)", avoid_direction, avoid_col)

    # Column 3: object count
    put(centers[2], "OBJECTS", str(obj_count), C_WHITE)

    # Draw vertical separator lines
    cv2.line(frame, (sec_w, y0 + 10), (sec_w, h - 10), C_ACCENT, 1)
    cv2.line(frame, (2 * sec_w, y0 + 10), (2 * sec_w, h - 10), C_ACCENT, 1)

    # Top-left title
    cv2.putText(frame, "VISION PIPELINE",
                (10, 30),
                cv2.FONT_HERSHEY_DUPLEX,
                0.6, C_ACCENT, 1, cv2.LINE_AA)

    return frame


def compute_display_size(src_w, src_h):
    # Resize the video frame to fit nicely on screen
    if src_w <= MAX_DISPLAY_W and src_h <= MAX_DISPLAY_H:
        return src_w, src_h

    scale_w = MAX_DISPLAY_W / src_w
    scale_h = MAX_DISPLAY_H / src_h
    scale = min(scale_w, scale_h)

    new_w = int(src_w * scale)
    new_h = int(src_h * scale)

    if new_w < MIN_DISPLAY_W or new_h < MIN_DISPLAY_H:
        scale_w_min = MIN_DISPLAY_W / src_w
        scale_h_min = MIN_DISPLAY_H / src_h
        scale = max(scale_w_min, scale_h_min)
        new_w = int(src_w * scale)
        new_h = int(src_h * scale)

    new_w = min(new_w, MAX_DISPLAY_W)
    new_h = min(new_h, MAX_DISPLAY_H)

    return new_w, new_h


def get_avoid_direction(boxes, frame_width, frame_height):
    # Figure out which side has more dangerous objects, so we can avoid that side
    left_threats = 0
    right_threats = 0

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bh = y2 - y1
        cx = (x1 + x2) // 2

        # Only consider large objects in the central area
        if bh > frame_height * 0.25 and frame_width * 0.25 <= cx <= frame_width * 0.75:
            if cx < frame_width * 0.5:
                left_threats += 1
            else:
                right_threats += 1

    if left_threats == 0 and right_threats == 0:
        return "GO"
    return "LEFT" if left_threats > right_threats else "RIGHT"


def run_pipeline(video_path, model):
    global FRAME_W, FRAME_H   # these are updated per video

    cap = cv2.VideoCapture(video_path)
    title = os.path.basename(video_path)

    if not cap.isOpened():
        print(f"  [!] Cannot open: {video_path}")
        return

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FRAME_W, FRAME_H = compute_display_size(src_w, src_h)
    print(f"  Processing: {title}  ({src_w}x{src_h} -> {FRAME_W}x{FRAME_H})")

    lane_detector = LaneDetector()
    frame_idx = 0
    last_boxes = None          # YOLO results from last processed frame
    last_avoid_dir = "GO"      # last avoidance suggestion
    last_obj_count = 0          # last count of important objects

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)

        #Lane detection
        left_line, right_line, lane_direction = lane_detector.process(frame)
        frame = draw_lanes(frame, left_line, right_line)

        #YOLO object detection (skip some frames for speed) ----
        if frame_idx % YOLO_SKIP_FRAMES == 0:
            results = model(frame, conf=CONF_THRESH, verbose=False)[0]
            last_boxes = results.boxes if results.boxes else None
            last_obj_count = 0
            if last_boxes:
                for box in last_boxes:
                    name = model.names[int(box.cls[0])]
                    if name in IMPORTANT_CLASSES:
                        last_obj_count += 1
                last_avoid_dir = get_avoid_direction(last_boxes, FRAME_W, FRAME_H)
            else:
                last_avoid_dir = "GO"

        # Draw YOLO detections if we have any
        if last_boxes:
            frame = draw_detections(frame, last_boxes, model.names)

        # Draw the heads-up display (bottom bar)
        frame = draw_hud(frame, lane_direction, last_avoid_dir, last_obj_count)
        frame_idx += 1

        # Show the result
        cv2.imshow(f"Vision Pipeline — {title}", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    print("Loading YOLOv8 model ...")
    model = YOLO('yolov8n.pt')     # small, fast YOLO model

    # Find all video files in the folder
    exts = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    videos = []
    for ext in exts:
        videos.extend(glob.glob(os.path.join(FOLDER_PATH, ext)))

    if not videos:
        print(f"No videos found in:\n  {FOLDER_PATH}")
        return

    print(f"Found {len(videos)} video(s).\n")
    for v in videos:
        run_pipeline(v, model)

    print("Done.")


if __name__ == "__main__":
    main()