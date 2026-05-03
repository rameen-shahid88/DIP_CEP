import cv2
import os
import glob
from ultralytics import YOLO

FOLDER_PATH = r"C:\Users\Rameen Shahid\Downloads\DIP Project Videos"
CONF_THRESH = 0.4
IMPORTANT = ['car', 'person', 'bicycle', 'bench',
             'chair', 'potted plant', 'trash can', 'wall', 'fence', 'curb', 'cat']

model = YOLO('yolov8n.pt')

video_exts = ['*.mp4', '*.avi', '*.mov', '*.mkv']
videos = []
for ext in video_exts:
    videos.extend(glob.glob(os.path.join(FOLDER_PATH, ext)))

for video_file in videos:
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        continue

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONF_THRESH, verbose=False)[0]
        stop = False
        h, w = frame.shape[:2]

        if results.boxes:
            for box in results.boxes:
                name = model.names[int(box.cls[0])]
                if name not in IMPORTANT:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                height = y2 - y1
                center_x = (x1 + x2) // 2
                if height > h * 0.25 and w*0.25 <= center_x <= w*0.75:
                    stop = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                cv2.putText(frame, name, (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        decision = "STOP" if stop else "GO"
        cv2.putText(frame, f"Decision: {decision}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        cv2.imshow(os.path.basename(video_file), frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()
        elif key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

cv2.destroyAllWindows()