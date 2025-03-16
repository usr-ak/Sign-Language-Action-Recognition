import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pyttsx3
import threading
from PyQt6.QtWidgets import QApplication, QLabel, QSlider, QVBoxLayout, QHBoxLayout, QWidget, QSpacerItem, QSizePolicy
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont

# Initialize Mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Initialize Text-to-Speech
engine = pyttsx3.init()

# Load Model
model = load_model(r"D:\P8\SL proj\code\no face\model_15na.h5")
actions = np.array(['no action', 'ready', 'hello', 'good', 'morning', 'whats up', 'how', 'you', 'i', 'fine', 'nice', 'meet', 'ok', 'thank you', 'see you later', 'good bye'])

sequence, sentence, predictions = [], [], []
threshold = 0.80
volume = 50
frame_count = 0
collecting_frames = False
hands_were_lowered = True
last_spoken_action = ""


class SignLanguageApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Sign Language to Text & Speech")
        self.showMaximized()

        # Main Heading
        self.main_heading = QLabel("Action Recognition", self)
        self.main_heading.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        self.main_heading.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Subheading
        self.sub_heading = QLabel("American Sign Language to Text and Speech", self)
        self.sub_heading.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        self.sub_heading.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Video Display
        self.video_label = QLabel(self)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumHeight(600)

        # Predicted Action & Sentence Labels
        self.predicted_action_label = QLabel("Predicted Action: ", self)
        self.predicted_action_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.predicted_action_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.predicted_action_label.setContentsMargins(65, 25, 5, 0)

        self.sentence_label = QLabel("Sentence: ", self)
        self.sentence_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.sentence_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.sentence_label.setContentsMargins(65, 15, 5, 0)

        # Threshold Control 
        self.threshold_label = QLabel(f"Threshold: {threshold:.2f}", self)
        self.threshold_slider = QSlider(Qt.Orientation.Vertical, self)
        self.threshold_slider.setMinimum(50)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(int(threshold * 100))
        self.threshold_slider.setTickPosition(QSlider.TickPosition.TicksBothSides)
        self.threshold_slider.setTickInterval(10)
        self.threshold_slider.setFixedSize(40, 200)
        self.threshold_slider.valueChanged.connect(self.update_threshold)

        # Volume Control
        self.volume_label = QLabel(f"Volume: {volume}%", self)
        self.volume_slider = QSlider(Qt.Orientation.Vertical, self)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(volume)
        self.volume_slider.setTickPosition(QSlider.TickPosition.TicksBothSides)
        self.volume_slider.setTickInterval(10)
        self.volume_slider.setFixedSize(40, 200)
        self.volume_slider.valueChanged.connect(self.update_volume)

        # Layouts
        self.video_layout = QVBoxLayout()
        self.video_layout.addWidget(self.main_heading)
        self.video_layout.addWidget(self.sub_heading)
        self.video_layout.addWidget(self.video_label)
        self.video_layout.addWidget(self.predicted_action_label)
        self.video_layout.addWidget(self.sentence_label)
        self.video_layout.addStretch(1)

        # Controls Layout
        self.controls_layout = QVBoxLayout()
        self.controls_layout.addStretch(1)
        self.controls_layout.addWidget(self.threshold_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.controls_layout.addWidget(self.threshold_slider, alignment=Qt.AlignmentFlag.AlignCenter)
        self.controls_layout.addSpacing(30)
        self.controls_layout.addWidget(self.volume_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.controls_layout.addWidget(self.volume_slider, alignment=Qt.AlignmentFlag.AlignCenter)
        self.controls_layout.addStretch(1)
        self.controls_layout.setContentsMargins(20, 0, 50, 0)

        self.main_layout = QHBoxLayout()
        self.main_layout.addLayout(self.video_layout)
        self.main_layout.addLayout(self.controls_layout)
        self.setLayout(self.main_layout)

        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)

        # Mediapipe model
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def update_frame(self):
        global frame_count, collecting_frames, hands_were_lowered, sentence, last_spoken_action

        ret, frame = self.cap.read()
        if not ret:
            return
        
        image, results = self.mediapipe_detection(frame)

        hands_detected = results.left_hand_landmarks or results.right_hand_landmarks

        if hands_detected and not collecting_frames and hands_were_lowered:
            collecting_frames = True
            frame_count = 0
            sequence.clear()
            hands_were_lowered = False

        if not hands_detected:
            hands_were_lowered = True

        if collecting_frames:
            keypoints = self.extract_keypoints(results)
            sequence.append(keypoints)
            frame_count += 1

            if frame_count >= 37:
                collecting_frames = False

        if len(sequence) == 37:
            input_data = sequence[7:]  
            res = model.predict(np.expand_dims(input_data, axis=0))[0]
            predicted_action = actions[np.argmax(res)]
            predictions.append(np.argmax(res))

            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    if predicted_action != "no action":
                        if len(sentence) == 0 or predicted_action != sentence[-1]:
                            sentence.append(predicted_action)

                        def speak(action):
                            engine.setProperty("volume", volume / 100.0)
                            engine.say(action)
                            engine.runAndWait()

                        if predicted_action != last_spoken_action:
                            last_spoken_action = predicted_action
                            threading.Thread(target=speak, args=(predicted_action,), daemon=True).start()

            if len(sentence) > 5:
                sentence = sentence[-5:]

            self.predicted_action_label.setText(f"Predicted Action: {predicted_action}")
            self.sentence_label.setText(f"Sentence: {' '.join(sentence)}")

        q_image = QImage(image.data, image.shape[1], image.shape[0], QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)

    def mediapipe_detection(self, image):
        """Perform pose and hand landmark detection"""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def extract_keypoints(self, results):
        """Extract keypoints from Mediapipe results"""
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, lh, rh])

    def update_threshold(self, value):
        global threshold
        threshold = value / 100
        self.threshold_label.setText(f"Threshold: {threshold:.2f}")

    def update_volume(self, value):
        global volume
        volume = value
        self.volume_label.setText(f"Volume: {volume}%")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Q:
            self.close()
        elif event.key() == Qt.Key.Key_F:
            if self.isFullScreen():
                self.showMaximized()
            else:
                self.showFullScreen()

    def closeEvent(self, event):
        self.cap.release()
        cv2.destroyAllWindows()
        self.holistic.close()
        event.accept()


app = QApplication([])
window = SignLanguageApp()
window.show()
app.exec()