
import cv2
import mediapipe as mp
import joblib
import pandas as pd
from collections import deque, Counter

# 1. Load the trained Random Forest model
model = joblib.load("isl_sign_model.pkl")

# 2. Create feature column names (21 landmarks x 3 coordinates)
column_names = [
    name
    for i in range(21)
    for name in (f"x{i}", f"y{i}", f"z{i}")
]

# 3. Store the last 7 predictions for smoothing
recent_predictions = deque(maxlen=7)

# 4. Configure MediaPipe Hand Landmarker
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

# 5. Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    raise SystemExit

timestamp_ms = 0

try:
    # 6. Create the hand detector
    with HandLandmarker.create_from_options(options) as detector:

        while cap.isOpened():

            # Read a frame from the webcam
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read webcam frame")
                break

            # Mirror the webcam image
            frame = cv2.flip(frame, 1)

            # Convert BGR image to RGB
            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            # Convert the frame to a MediaPipe image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Use increasing timestamps for video mode
            timestamp_ms += 33

            # Detect hand landmarks
            result = detector.detect_for_video(
                mp_image, timestamp_ms
            )

            prediction_text = "Show your hand"

            # 7. Process the detected hand
            if result.hand_landmarks:

                hand = result.hand_landmarks[0]
                features = []

                # Extract x, y, z from each of the 21 landmarks
                for landmark in hand:
                    features.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ])

                # Create a DataFrame with the training column names
                input_data = pd.DataFrame(
                    [features],
                    columns=column_names
                )

                # Predict the sign
                prediction = model.predict(input_data)[0]

                # Store the prediction for smoothing
                recent_predictions.append(prediction)

                # Find the most frequent recent prediction
                most_common = Counter(
                    recent_predictions
                ).most_common(1)

                prediction_text = (
                    f"Prediction: {most_common[0][0]}"
                )

                # Draw green landmark dots
                for landmark in hand:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

            else:
                # Clear old predictions when no hand is detected
                recent_predictions.clear()

            # 8. Display the prediction on the webcam
            cv2.putText(
                frame,
                prediction_text,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Press Q to quit",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow("ISL Sign Recognition", frame)

            # Press Q to exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    # 9. Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

print("Sign recognition stopped.")
