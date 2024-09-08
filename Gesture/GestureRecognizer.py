import cv2
import mediapipe as mp
import numpy as np

class GestureRecognizer:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        keypoints = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    keypoints.append([landmark.x, landmark.y, landmark.z])
        
        return np.array(keypoints).flatten()

    def visualize(self, frame, keypoints):
        if len(keypoints) > 0:
            for landmark in keypoints:
                x, y, _ = landmark
                h, w, _ = frame.shape
                x, y = int(x * w), int(y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        return frame