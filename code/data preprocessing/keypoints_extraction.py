import os
import numpy as np
import cv2
import concurrent.futures
from mediapipe import solutions as mp

# Initialize Mediapipe
mp_holistic = mp.holistic.Holistic
actions = np.array(['no action', 'ready', 'hello', 'good', 'morning', 'whats up', 'how', 'you', 'i', 'fine', 'nice', 'meet', 'ok', 'thank you', 'see you later', 'good bye'])
KEYPOINTS_PATH = r'D:\P8\others\data\MP_DataNF'
VIDEO_PATH = r'D:\P8\others\data\SLVideo'
no_sequences = 110
sequence_length = 30

# Create directory structure for saving keypoints
for action in actions:
    for sequence in range(no_sequences):
        os.makedirs(os.path.join(KEYPOINTS_PATH, action, str(sequence)), exist_ok=True)

# Define keypoint extraction function
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, lh, rh])

# Process videos and save keypoints for each video using parallelization
def process_video(action, sequence, video_path):
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    with mp_holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_num >= sequence_length:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            keypoints = extract_keypoints(results)
            npy_path = os.path.join(KEYPOINTS_PATH, action, str(sequence), f"{frame_num}.npy")
            np.save(npy_path, keypoints)
            frame_num += 1
    cap.release()

# Using ThreadPoolExecutor for parallel processing
with concurrent.futures.ThreadPoolExecutor() as executor:
    for action in actions:
        for sequence in range(no_sequences):
            video_path = os.path.join(VIDEO_PATH, action, str(sequence), 'video.avi')
            if not os.path.exists(video_path):
                print(f"Video {video_path} not found. Skipping.")
                continue
            # Submit video processing tasks to the executor
            executor.submit(process_video, action, sequence, video_path)