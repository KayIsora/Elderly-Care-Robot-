# import cv2
# import mediapipe as mp
# import numpy as np
# import tensorflow as tf
# import time
# import matplotlib.pyplot as plt

# # ======== FUNCTION TO EXTRACT 63 FEATURES (OpenPose-style) =========
# def extract_openpose_style_landmarks(results, width, height):
#     lm = results.pose_landmarks.landmark
#     coords = []

#     def get_xyc(idx):
#         return [lm[idx].x * width, lm[idx].y * height, lm[idx].visibility]

#     # 0: Nose
#     coords += get_xyc(0)

#     # 1: Neck (average of left/right shoulder)
#     neck_x = (lm[11].x + lm[12].x) / 2.0 * width
#     neck_y = (lm[11].y + lm[12].y) / 2.0 * height
#     neck_c = (lm[11].visibility + lm[12].visibility) / 2.0
#     coords += [neck_x, neck_y, neck_c]

#     # 2-20: RShoulder → LAnkle + upper body keypoints
#     key_ids = [12, 14, 16, 11, 13, 15, 24, 26, 28, 23, 25, 27, 5, 6, 7, 2, 3, 4, 8]
#     for i in key_ids:
#         coords += get_xyc(i)

#     return coords  # 21 keypoints × 3 values = 63

# # ======== INITIALIZE MODEL & MEDIAPIPE =========
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(static_image_mode=False)
# mp_drawing = mp.solutions.drawing_utils  # To draw keypoints

# model = tf.keras.models.load_model('final_lstm_model.h5')

# # ======== INITIALIZE VIDEO =========
# cap = cv2.VideoCapture('/home/jetson/Desktop/Quy/ab/test.mp4')  # Replace with your own video path
# fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
# fall_probs = []
# window_data = []

# print("Starting real-time fall detection...")

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     h, w, _ = frame.shape
#     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = pose.process(image)

#     if results.pose_landmarks:
#         # Extract and normalize keypoints
#         filtered = extract_openpose_style_landmarks(results, w, h)
#         for i in range(0, 63, 3):
#             filtered[i] /= w
#             filtered[i + 1] /= h
#         window_data.append(filtered)

#         # Predict
#         latest_frame = np.array([window_data[-1]]).reshape((1, 1, 63))
#         fall_prob = model.predict(latest_frame, verbose=0)[0][0]
#         fall_probs.append(fall_prob)

#         label = "Fall Probability: {:.2f}".format(fall_prob)
#         color = (0, 0, 255) if fall_prob > 0.5 else (0, 255, 0)
#     else:
#         label = "No person detected"
#         color = (255, 255, 0)
#         fall_prob = None

#     # ======== DRAW KEYPOINTS =========
#     if results.pose_landmarks:
#         mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#     # Show label
#     cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
#     cv2.imshow('Fall Detection', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# # ======== PLOT FALL PROBABILITY OVER TIME =========
# if fall_probs:
#     times = np.arange(0, len(fall_probs)) / fps
#     plt.plot(times, fall_probs)
#     plt.xlabel("Time (s)")
#     plt.ylabel("Fall Probability")
#     plt.ylim(-0.05, 1.05)
#     plt.title("Fall Probability Over Time")
#     plt.grid()
#     plt.savefig("fall_probability_plot.png")
#     print("Fall probability plot saved to 'fall_probability_plot.png'")
# else:
#     print("No fall probability data available to plot.")

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

# ======== FUNCTION TO EXTRACT 63 FEATURES (OpenPose-style) =========
def extract_openpose_style_landmarks(results, width, height):
    lm = results.pose_landmarks.landmark
    coords = []

    def get_xyc(idx):
        return [lm[idx].x * width, lm[idx].y * height, lm[idx].visibility]

    # 0: Nose
    coords += get_xyc(0)

    # 1: Neck (average of left/right shoulder)
    neck_x = (lm[11].x + lm[12].x) / 2.0 * width
    neck_y = (lm[11].y + lm[12].y) / 2.0 * height
    neck_c = (lm[11].visibility + lm[12].visibility) / 2.0
    coords += [neck_x, neck_y, neck_c]

    # 2-20: RShoulder → LAnkle + upper body keypoints
    key_ids = [12, 14, 16, 11, 13, 15, 24, 26, 28, 23, 25, 27, 5, 6, 7, 2, 3, 4, 8]
    for i in key_ids:
        coords += get_xyc(i)

    return coords  # 21 keypoints × 3 values = 63

# ======== INITIALIZE MODEL & MEDIAPIPE =========
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils  # To draw keypoints

model = tf.keras.models.load_model('final_lstm_model.h5')

# ======== INITIALIZE VIDEO =========
cap = cv2.VideoCapture('video (10).avi')  # Replace with your own video path
fps_video = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
fall_probs = []
window_data = []

# Variables to store time measurements
mediapipe_processing_times = []
prediction_times = []

print("Starting real-time fall detection...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- Measure MediaPipe processing time ---
    start_mediapipe_time = time.time()
    results = pose.process(image)
    end_mediapipe_time = time.time()
    mediapipe_processing_times.append(end_mediapipe_time - start_mediapipe_time)
    # --- End MediaPipe processing time measurement ---

    if results.pose_landmarks:
        # Extract and normalize keypoints
        filtered = extract_openpose_style_landmarks(results, w, h)
        for i in range(0, 63, 3):
            filtered[i] /= w
            filtered[i + 1] /= h
        window_data.append(filtered)

        # Predict
        # --- Measure prediction time ---
        start_prediction_time = time.time()
        latest_frame = np.array([window_data[-1]]).reshape((1, 1, 63))
        fall_prob = model.predict(latest_frame, verbose=0)[0][0]
        end_prediction_time = time.time()
        prediction_times.append(end_prediction_time - start_prediction_time)
        # --- End prediction time measurement ---

        fall_probs.append(fall_prob)

        label = "Fall Probability: {:.2f}".format(fall_prob)
        color = (0, 0, 255) if fall_prob > 0.5 else (0, 255, 0)
    else:
        label = "No person detected"
        color = (255, 255, 0)
        fall_prob = None
        # Append 0 to times if no person detected, to keep lists aligned with frames
        mediapipe_processing_times.append(0)
        prediction_times.append(0)

    # ======== DRAW KEYPOINTS =========
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show label
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    # cv2.imshow('Fall Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ======== CALCULATE AND PRINT AVERAGE FPS ========
if mediapipe_processing_times:
    avg_mediapipe_time = np.mean(mediapipe_processing_times)
    mediapipe_fps = 1 / avg_mediapipe_time if avg_mediapipe_time > 0 else 0
    print(f"\nAverage MediaPipe Processing FPS: {mediapipe_fps:.2f}")
else:
    print("\nNo MediaPipe processing data to calculate FPS.")

if prediction_times:
    avg_prediction_time = np.mean(prediction_times)
    prediction_fps = 1 / avg_prediction_time if avg_prediction_time > 0 else 0
    print(f"Average Model Prediction FPS: {prediction_fps:.2f}")
else:
    print("No model prediction data to calculate FPS.")

# ======== PLOT FALL PROBABILITY OVER TIME =========
if fall_probs:
    times = np.arange(0, len(fall_probs)) / fps_video
    plt.plot(times, fall_probs)
    plt.xlabel("Time (s)")
    plt.ylabel("Fall Probability")
    plt.ylim(-0.05, 1.05)
    plt.title("Fall Probability Over Time")
    plt.grid()
    plt.savefig("fall_probability_plot.png")
    print("Fall probability plot saved to 'fall_probability_plot.png'")
else:
    print("No fall probability data available to plot.")
