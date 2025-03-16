import cv2
import os
import time
import numpy as np
import mediapipe as mp

# Initialize Mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Define functions
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
    )
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
    )
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )
    
# Function to wait with pause functionality
def wait_with_pause(wait_time, cap, action, sequence):
    while True:
        start_time = time.time()
        paused = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            elapsed_time = time.time() - start_time
            remaining_time = max(0, int(wait_time - elapsed_time))

            # Display countdown on the raw frame
            cv2.putText(frame, f"Starting sequence {sequence + 1}/{no_sequences} for '{action}'", (50, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Starting in {remaining_time}s... Press 'P' to pause", (50, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('OpenCV Feed', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

            if key == ord('p'):
                paused = True
                while paused:
                    ret, frame = cap.read()
                    cv2.putText(frame, "Paused... Press 'P' to resume", (50, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('p'):
                        paused = False
                        break  # Restart countdown

                break  # Restart wait time

            if elapsed_time >= wait_time:
                return  # Continue execution

# Data collection parameters
VIDEO_PATH = os.path.join('SLVideo')
actions = np.array(['no action', 'ready', 'hello', 'good', 'morning', 'whats up', 'how', 'you', 'i', 'fine', 'nice', 'meet', 'ok', 'thank you', 'see you later', 'good bye'])
no_sequences = int(input("Enter no. of sequences for actions : "))
sequence_length = 30

# Create folders
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(VIDEO_PATH, action, str(sequence)))
        except:
            pass

# Collect data and store video
cap = cv2.VideoCapture(0)
cv2.namedWindow("OpenCV Feed", cv2.WINDOW_NORMAL)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.putText(frame, f"Press 'S' to start capturing '{action}'", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()
            if key == ord('s'):
                break

        # 3-second wait before starting
        wait_with_pause(3, cap, action, 0)

        print(f"Starting collection for '{action}'...")
        for sequence in range(no_sequences):
            video_writer = cv2.VideoWriter(os.path.join(VIDEO_PATH, action, str(sequence), 'video.avi'),
                                           cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))
            
            start_time = time.time()
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    break

                # Save the original frame without landmarks or text for the video file
                original_frame = frame.copy()

                # Process the frame for display
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)
                
                # Add progress text on the display image
                cv2.putText(image, f"Capturing '{action}', Sequence {sequence + 1}/{no_sequences}, Frame {frame_num + 1}", 
                            (20, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                # Write the original (unmodified) frame to the video file
                video_writer.write(original_frame)

                # Display the annotated image
                cv2.imshow('OpenCV Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    print("Exiting...")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

            video_writer.release()
            end_time = time.time()
            sequence_time = end_time - start_time
            print(f"Time taken for {sequence_length} frames in sequence {sequence + 1}: {sequence_time:.2f} seconds")

            if sequence < no_sequences - 1:
                wait_with_pause(3, cap, action, sequence + 1)

        print(f"Finished collecting all sequences for action: {action}")

cap.release()
cv2.destroyAllWindows()