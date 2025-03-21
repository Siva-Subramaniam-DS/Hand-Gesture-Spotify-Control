import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import random
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Initialize MediaPipe Hands and Drawing modules
mp_hands = mp.solutions.hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Gesture mapping (example actions)
gesture_mapping = {
    "Fist": "Pause/Play",
    "Open Palm": "Activate Control",
    "Thumbs Up": "Volume Up",
    "Thumbs Down": "Volume Down",
    "Swipe Left": "Previous Track",
    "Swipe Right": "Next Track"
}

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.activation_flag = False
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = mp_hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                
                # Extract landmarks
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                
                # Random gesture simulation (replace with ML model prediction)
                detected_gesture = random.choice(list(gesture_mapping.keys()))
                
                # Activation condition
                if detected_gesture == "Open Palm":
                    self.activation_flag = True
                    st.write("Open Palm detected! Activation enabled.")
                
                if self.activation_flag and detected_gesture in gesture_mapping:
                    action_code = gesture_mapping[detected_gesture]
                    st.write(f"Detected Gesture: {detected_gesture} → Action: {action_code}")
        
        return img

# Streamlit UI
st.title("Hand Gesture-Controlled Spotify")
st.write("Use hand gestures to control Spotify playback.")

webrtc_streamer(key="gesture-control", video_transformer_factory=VideoTransformer)