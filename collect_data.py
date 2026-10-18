
import cv2
import mediapipe as mp
import csv
import os

# 1. File where our dataset will be saved
DATA_FILE = "sign_data.csv"

# 2. Keyboard keys mapped to sign labels
LABELS = {
    "1": "ISL_SIGN_1",
    "2": "ISL_SIGN_2",
    "3": "ISL_SIGN_3"
}

# 3. Create CSV column names for 21 landmarks
header = []

for i in range(21):
    header.extend([f"x{i}", f"y{i}", f"z{i}"])

header.append("label")

# 4. Add the header only if the CSV file is new
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)

# 5. Configure the MediaPipe hand detector
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

# 6. Hand skeleton connections
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# 7. Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    raise SystemExit

timestamp_ms = 0

try:
    with HandLandmarker.create_from_options(options) as detector:

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("Could not read webcam frame")
                break

            # Mirror the webcam image
            frame = cv2.flip(frame, 1)

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            # Create a MediaPipe image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Detect hand landmarks
            timestamp_ms += 33
            result = detector.detect_for_video(
                mp_image, timestamp_ms
            )

            points = []
            features = []

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]

                for landmark in hand:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])

                    points.append((x, y))

                    features.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ])

                # Draw the landmarks
                for x, y in points:
                    cv2.circle(
                        frame, (x, y), 5, (0, 255, 0), -1
                    )

                # Draw the skeleton
                for start_idx, end_idx in connections:
                    cv2.line(
                        frame,
                        points[start_idx],
                        points[end_idx],
                        (255, 0, 0),
                        2
                    )

            # Show collection instructions
            cv2.putText(
                frame,
                "Press 1, 2, 3 to save a sign sample",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Press Q to quit",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.imshow("ISL Data Collection", frame)

            # Read the pressed key
            key = cv2.waitKey(1) & 0xFF

            # Save features with the selected label
            if chr(key) in LABELS and len(features) == 63:
                label = LABELS[chr(key)]

                with open(
                    DATA_FILE, "a", newline="", encoding="utf-8"
                ) as file:
                    writer = csv.writer(file)
                    writer.writerow(features + [label])

                print(f"Saved one sample: {label}")

            # Quit when Q is pressed
            if key == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()

print("Data collection stopped.")
