Fall Detection:
1. Use OpenPose to extract keypoints (body landmarks). 
        Note: Keypoints are already saved in a CSV file, so no need to extract them again.
2. Train the model using CNN and LSTM.
4. Inference on Jetson Nano:
→ Use MediaPipe Pose for real-time keypoint extraction
→ Pass keypoints to the trained CNN model for fall prediction.

