import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('path/to/your/gesture_model.h5')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

def preprocess_gesture(image, landmarks):
    gesture_data = np.array([landmark for landmark in landmarks])
    return gesture_data

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            gesture_data = preprocess_gesture(image, landmarks)
            gesture_data = np.expand_dims(gesture_data, axis=0)  # Add batch dimension
            
            prediction = model.predict(gesture_data)
            predicted_class = np.argmax(prediction, axis=1)
            
            gesture_to_sentence = {0: "Hello", 1: "Goodbye"}  # Example mapping
            sentence = gesture_to_sentence.get(predicted_class[0], "Unknown")
            
            cv2.putText(image, sentence, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Hand Gesture Recognition', image)
    if cv2.waitKey(5) & 0xFF == 27:  # Exit on ESC
        break

cap.release()
cv2.destroyAllWindows()