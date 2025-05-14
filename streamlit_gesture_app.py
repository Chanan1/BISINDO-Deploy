
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
from PIL import Image

# Load model
model = load_model("model.h5")  # Pastikan file ini ada di folder yang sama

# Load Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Daftar label (huruf a-z)
labels = [chr(i) for i in range(97, 123)]  # ['a', 'b', ..., 'z']

# Fungsi menggambar landmark ke citra 256x256
def draw_landmarks_to_canvas(landmarks, image_shape):
    canvas = np.zeros((256, 256), dtype=np.uint8)
    for lm in landmarks:
        x = int(lm.x * 256)
        y = int(lm.y * 256)
        cv2.circle(canvas, (x, y), 3, 255, -1)
    return canvas

# Streamlit UI
st.title("Real-time ASL Alphabet Prediction")
frame_window = st.image([])

run = st.checkbox('Start Webcam')

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Gagal membuka webcam.")
        break

    # Flip & convert BGR to RGB
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe processing
    results = hands.process(rgb)

    pred_label = "-"
    if results.multi_hand_landmarks:
        canvas = np.zeros((256, 256), dtype=np.uint8)
        for hand_landmarks in results.multi_hand_landmarks:
            canvas = draw_landmarks_to_canvas(hand_landmarks.landmark, frame.shape[:2])
        
        # Normalisasi & prediksi
        input_image = canvas / 255.0
        input_image = input_image.reshape(1, 256, 256, 1)  # sesuai model input
        pred = model.predict(input_image)
        pred_label = labels[np.argmax(pred)]

    # Tampilkan label prediksi
    cv2.putText(frame, f'Prediction: {pred_label}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tampilkan di Streamlit
    frame_window.image(frame, channels="BGR")

cap.release()
