
import os
import cv2
import time
import numpy as np
import mediapipe as mp
import joblib

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "drowsiness_model.pkl")
TASK_PATH = os.path.join(BASE_DIR, "face_landmarker.task")

# 2. Load trained ML model
model = joblib.load(MODEL_PATH)

# 3. MediaPipe face landmark settings
base_options = python.BaseOptions(
    model_asset_path=TASK_PATH
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.20
CLOSED_SECONDS = 2.0


# 4. Calculate Eye Aspect Ratio
def calculate_ear(landmarks, indices, width, height):
    points = []

    for index in indices:
        p = landmarks[index]
        points.append(
            np.array([p.x * width, p.y * height])
        )

    vertical_1 = np.linalg.norm(points[1] - points[5])
    vertical_2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])

    if horizontal == 0:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


# 5. Extract eye crop for ML prediction
def get_eye_crop(frame, landmarks, indices):
    height, width = frame.shape[:2]
    points = []

    for index in indices:
        p = landmarks[index]
        points.append(
            (int(p.x * width), int(p.y * height))
        )

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    pad_x = max(5, int((max(xs) - min(xs)) * 0.5))
    pad_y = max(5, int((max(ys) - min(ys)) * 1.0))

    x1 = max(0, min(xs) - pad_x)
    x2 = min(width, max(xs) + pad_x)
    y1 = max(0, min(ys) - pad_y)
    y2 = min(height, max(ys) + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]


# 6. Predict eye state using the trained model
def predict_eye(eye_crop):
    if eye_crop is None or eye_crop.size == 0:
        return "UNKNOWN"

    gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (24, 24))
    gray = gray.astype(np.float32) / 255.0

    features = gray.flatten().reshape(1, -1)
    prediction = model.predict(features)[0]

    return "OPEN" if prediction == 1 else "CLOSED"


# 7. Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam could not be opened.")
    raise SystemExit

closed_start = None
last_alert = 0
start_time = time.monotonic()

print("Driver Drowsiness Alert System Started")
print("Press Q to quit.")

try:
    with vision.FaceLandmarker.create_from_options(options) as landmarker:

        while True:
            success, frame = cap.read()

            if not success:
                print("ERROR: Cannot read webcam.")
                break

            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = int(
                (time.monotonic() - start_time) * 1000
            )

            result = landmarker.detect_for_video(
                mp_image, timestamp_ms
            )

            status = "Face not detected"
            ear_text = ""

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]

                # Calculate EAR for both eyes
                left_ear = calculate_ear(
                    landmarks, LEFT_EYE, width, height
                )

                right_ear = calculate_ear(
                    landmarks, RIGHT_EYE, width, height
                )

                ear_text = (
                    f"EAR L:{left_ear:.2f} "
                    f"R:{right_ear:.2f}"
                )

                # Crop both eyes
                left_crop = get_eye_crop(
                    frame, landmarks, LEFT_EYE
                )

                right_crop = get_eye_crop(
                    frame, landmarks, RIGHT_EYE
                )

                # ML model predictions (for comparison)
                left_ml = predict_eye(left_crop)
                right_ml = predict_eye(right_crop)

                # EAR-based closure detection
                if (
                    left_ear < EAR_THRESHOLD
                    and right_ear < EAR_THRESHOLD
                ):
                    status = "EYES CLOSED"

                    if closed_start is None:
                        closed_start = time.monotonic()

                    elapsed = time.monotonic() - closed_start

                    if elapsed >= CLOSED_SECONDS:
                        status = "DROWSINESS ALERT!"

                        if time.monotonic() - last_alert >= 2:
                            try:
                                import winsound
                                winsound.Beep(1500, 400)
                            except (ImportError, RuntimeError):
                                pass

                            last_alert = time.monotonic()
                else:
                    closed_start = None
                    status = "EYES OPEN"

                # Draw eye landmark points
                for index in LEFT_EYE + RIGHT_EYE:
                    x = int(landmarks[index].x * width)
                    y = int(landmarks[index].y * height)

                    cv2.circle(
                        frame, (x, y), 2, (0, 255, 0), -1
                    )

                cv2.putText(
                    frame,
                    f"ML: L={left_ml} R={right_ml}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2
                )

            color = (
                (0, 0, 255)
                if "ALERT" in status
                else (0, 255, 0)
            )

            cv2.putText(
                frame,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            if ear_text:
                cv2.putText(
                    frame,
                    ear_text,
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

            cv2.imshow(
                "Driver Drowsiness Alert System",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
