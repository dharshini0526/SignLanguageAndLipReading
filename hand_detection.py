
import cv2
import mediapipe as mp

# 1. Configure the MediaPipe hand detector
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

# 2. Define connections between the 21 hand landmarks
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# 3. Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    cap.release()
    raise SystemExit

# 4. Create the hand detector and process video
try:
    with HandLandmarker.create_from_options(options) as detector:
        timestamp_ms = 0

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read webcam frame")
                break

            # Mirror the video for a natural webcam view
            frame = cv2.flip(frame, 1)

            # Convert BGR (OpenCV) to RGB (MediaPipe)
            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            # Create a MediaPipe image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Use increasing timestamps for video frames
            timestamp_ms += 33

            # Detect hands
            result = detector.detect_for_video(
                mp_image, timestamp_ms
            )

            # 5. Draw hand landmarks and connections
            if result.hand_landmarks:
                for hand in result.hand_landmarks:

                    points = []

                    # Convert normalized coordinates to pixels
                    for landmark in hand:
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        points.append((x, y))

                    # Draw green circles
                    for x, y in points:
                        cv2.circle(
                            frame,
                            (x, y),
                            5,
                            (0, 255, 0),
                            -1
                        )

                    # Draw blue lines
                    for start_idx, end_idx in connections:
                        cv2.line(
                            frame,
                            points[start_idx],
                            points[end_idx],
                            (255, 0, 0),
                            2
                        )

            # 6. Display the result
            cv2.imshow("Sign Language - Hand Detection", frame)

            # Press q to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    # 7. Release resources
    cap.release()
    cv2.destroyAllWindows()
