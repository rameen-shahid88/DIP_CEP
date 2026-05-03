import cv2
import numpy as np
import os
import glob
from collections import deque
from ultralytics import YOLO

FRAME_W         = 0
FRAME_H         = 0
MAX_DISPLAY_W   = 800
CONF_THRESH     = 0.4
HISTORY_LEN     = 12
SMOOTHING_ALPHA = 0.52
MAX_MISS_FRAMES  = 8
YOLO_SKIP_FRAMES = 3

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
        self.prev_left  = None
        self.prev_right = None
        self.left_miss  = 0
        self.right_miss = 0
        self.history    = deque(maxlen=HISTORY_LEN)

    def get_edges(self, frame):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray  = cv2.equalizeHist(gray)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        sigma = 0.33
        med   = np.median(blur)
        lo    = int(max(0,   (1.0 - sigma) * med))
        hi    = int(min(255, (1.0 + sigma) * med))
        return cv2.Canny(blur, lo, hi)

    def region_of_interest(self, edges):
        h, w      = edges.shape
        row_sums  = np.sum(edges, axis=1)
        threshold = w * 0.04 * 255
        horizon   = int(h * 0.45)
        for r in range(int(h * 0.3), int(h * 0.65)):
            if row_sums[r] > threshold:
                horizon = r
                break
        horizon = max(int(h * 0.35), min(horizon, int(h * 0.62)))
        mask    = np.zeros_like(edges)
        poly    = np.array([[
            (int(0.05 * w), h),
            (int(0.95 * w), h),
            (int(0.65 * w), horizon),
            (int(0.35 * w), horizon)
        ]])
        cv2.fillPoly(mask, poly, 255)
        return cv2.bitwise_and(edges, mask)

    def get_raw_lines(self, edges):
        h, w  = edges.shape
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50,
                                minLineLength=50, maxLineGap=100)
        left, right = [], []
        if lines is None:
            return left, right
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.5 or abs(slope) > 2.0:
                continue
            if slope < 0 and x1 < w * 0.55 and x2 < w * 0.55:
                left.append((x1, y1, x2, y2))
            elif slope > 0 and x1 > w * 0.45 and x2 > w * 0.45:
                right.append((x1, y1, x2, y2))
        return left, right

    def average_line(self, lines):
        if not lines:
            return None
        xs, ys, ws = [], [], []
        for x1, y1, x2, y2 in lines:
            length = np.hypot(x2 - x1, y2 - y1)
            xs    += [x1, x2]
            ys    += [y1, y2]
            ws    += [length, length]
        try:
            poly = np.polyfit(ys, xs, 1, w=ws)
        except np.linalg.LinAlgError:
            return None
        y_bot = FRAME_H
        y_top = int(FRAME_H * 0.60)
        return (int(np.polyval(poly, y_bot)), y_bot,
                int(np.polyval(poly, y_top)), y_top)

    def _smooth(self, new_line, prev_line, miss_count):
        if new_line is None:
            if miss_count >= MAX_MISS_FRAMES:
                return None
            return prev_line
        if prev_line is None:
            return new_line
        return tuple(
            int(SMOOTHING_ALPHA * n + (1.0 - SMOOTHING_ALPHA) * p)
            for n, p in zip(new_line, prev_line)
        )

    def get_direction(self, left_line, right_line):
        w      = FRAME_W
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
        edges               = self.get_edges(frame)
        roi                 = self.region_of_interest(edges)
        left_raw, right_raw = self.get_raw_lines(roi)
        raw_left            = self.average_line(left_raw)
        raw_right           = self.average_line(right_raw)

        self.left_miss  = 0 if raw_left  is not None else self.left_miss  + 1
        self.right_miss = 0 if raw_right is not None else self.right_miss + 1

        left_line  = self._smooth(raw_left,  self.prev_left,  self.left_miss)
        right_line = self._smooth(raw_right, self.prev_right, self.right_miss)

        if left_line  is not None: self.prev_left  = left_line
        if right_line is not None: self.prev_right = right_line

        direction = self.get_direction(left_line, right_line)
        self.history.append(direction)
        direction = max(set(self.history), key=self.history.count)

        return left_line, right_line, direction


def draw_lanes(frame, left_line, right_line):
    overlay = frame.copy()
    if left_line and right_line:
        pts = np.array([
            [left_line[0],  left_line[1]],
            [left_line[2],  left_line[3]],
            [right_line[2], right_line[3]],
            [right_line[0], right_line[1]]
        ], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], C_GREEN)
        frame = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)
    for line, colour in [(left_line, C_GREEN), (right_line, C_GREEN)]:
        if line:
            cv2.line(frame, (line[0], line[1]), (line[2], line[3]),
                     colour, 4, cv2.LINE_AA)
    return frame


def draw_detections(frame, boxes, names):
    h, w = frame.shape[:2]
    for box in boxes:
        name = names[int(box.cls[0])]
        if name not in IMPORTANT_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf            = float(box.conf[0])
        bh, cx          = y2 - y1, (x1 + x2) // 2
        is_threat       = (bh > h * 0.25 and w * 0.25 <= cx <= w * 0.75)
        colour          = C_RED if is_threat else C_ACCENT
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        label       = f"{name}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
        pad         = 4
        cv2.rectangle(frame, (x1, y1 - th - 2*pad), (x1 + tw + 2*pad, y1), colour, -1)
        cv2.putText(frame, label, (x1 + pad, y1 - pad),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, C_DARK, 1, cv2.LINE_AA)
    return frame


def draw_hud(frame, direction, decision, obj_count):
    h, w  = frame.shape[:2]
    bar_h = 72
    bar   = frame[h - bar_h:h, :].copy()
    cv2.rectangle(bar, (0, 0), (w, bar_h), C_DARK, -1)
    frame[h - bar_h:h, :] = cv2.addWeighted(bar, 0.72, frame[h - bar_h:h, :], 0.28, 0)
    cv2.line(frame, (0, h - bar_h), (w, h - bar_h), C_ACCENT, 1)

    dir_col = C_GREEN if direction == "STRAIGHT" else C_YELLOW
    cv2.putText(frame, "LANE",      (28, h - bar_h + 22), cv2.FONT_HERSHEY_DUPLEX, 0.45, C_ACCENT, 1, cv2.LINE_AA)
    cv2.putText(frame, direction,   (28, h - bar_h + 52), cv2.FONT_HERSHEY_DUPLEX, 0.85, dir_col,  2, cv2.LINE_AA)

    cv2.line(frame, (w//3, h - bar_h + 10), (w//3, h - 10), C_ACCENT, 1)

    dec_col = C_RED if decision == "STOP" else C_GREEN
    cv2.putText(frame, "DECISION",  (w//3 + 28, h - bar_h + 22), cv2.FONT_HERSHEY_DUPLEX, 0.45, C_ACCENT, 1, cv2.LINE_AA)
    cv2.putText(frame, decision,    (w//3 + 28, h - bar_h + 52), cv2.FONT_HERSHEY_DUPLEX, 0.95, dec_col,  2, cv2.LINE_AA)

    cv2.line(frame, (2*w//3, h - bar_h + 10), (2*w//3, h - 10), C_ACCENT, 1)

    cv2.putText(frame, "OBJECTS",       (2*w//3 + 28, h - bar_h + 22), cv2.FONT_HERSHEY_DUPLEX, 0.45, C_ACCENT, 1, cv2.LINE_AA)
    cv2.putText(frame, str(obj_count),  (2*w//3 + 28, h - bar_h + 52), cv2.FONT_HERSHEY_DUPLEX, 0.95, C_WHITE,  2, cv2.LINE_AA)

    cv2.putText(frame, "VISION PIPELINE", (14, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, C_ACCENT, 1, cv2.LINE_AA)
    return frame


def compute_display_size(src_w, src_h):
    if src_w <= MAX_DISPLAY_W:
        return src_w, src_h
    scale = MAX_DISPLAY_W / src_w
    return MAX_DISPLAY_W, int(src_h * scale)


def run_pipeline(video_path, model):
    global FRAME_W, FRAME_H

    cap   = cv2.VideoCapture(video_path)
    title = os.path.basename(video_path)

    if not cap.isOpened():
        print(f"  [!] Cannot open: {video_path}")
        return

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FRAME_W, FRAME_H = compute_display_size(src_w, src_h)
    print(f"  Processing: {title}  ({src_w}x{src_h} -> {FRAME_W}x{FRAME_H})")

    lane_detector  = LaneDetector()
    frame_idx      = 0
    last_boxes     = None
    last_stop      = False
    last_obj_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)

        left_line, right_line, direction = lane_detector.process(frame)
        frame = draw_lanes(frame, left_line, right_line)

        if frame_idx % YOLO_SKIP_FRAMES == 0:
            results        = model(frame, conf=CONF_THRESH, verbose=False)[0]
            last_boxes     = results.boxes if results.boxes else None
            last_stop      = False
            last_obj_count = 0
            if last_boxes:
                fh, fw = frame.shape[:2]
                for box in last_boxes:
                    name = model.names[int(box.cls[0])]
                    if name not in IMPORTANT_CLASSES:
                        continue
                    last_obj_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    bh, cx = y2 - y1, (x1 + x2) // 2
                    if bh > fh * 0.25 and fw * 0.25 <= cx <= fw * 0.75:
                        last_stop = True

        if last_boxes:
            frame = draw_detections(frame, last_boxes, model.names)

        decision = "STOP" if last_stop else "GO"
        frame    = draw_hud(frame, direction, decision, last_obj_count)
        frame_idx += 1

        cv2.imshow(f"Vision Pipeline — {title}", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    print("Loading YOLOv8 model ...")
    model  = YOLO('yolov8n.pt')

    exts   = ['*.mp4', '*.avi', '*.mov', '*.mkv']
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