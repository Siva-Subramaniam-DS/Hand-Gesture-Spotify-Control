# Hand Gesture Spotify Control

Control music playback with hand gestures. The project has two parts: a notebook that trains a CNN to recognise six hand gestures, and a Streamlit webcam demo that maps gestures to playback actions.

## Gestures and actions

| Gesture | Action |
|---|---|
| Open Palm | Activate control |
| Fist | Pause / Play |
| Thumbs Up | Volume up |
| Thumbs Down | Volume down |
| Swipe Left | Previous track |
| Swipe Right | Next track |

Control stays off until an Open Palm is seen, which avoids accidental triggers.

## Files

| File | Purpose |
|---|---|
| `Hand Gesture Recognition.ipynb` | Builds the dataset, trains and evaluates the CNN (64x64 grayscale input) |
| `hand_gesture_model.h5` | The trained Keras model |
| `main.py` | Streamlit app: webcam stream, MediaPipe hand landmarks, gesture to action mapping |

## Requirements

```bash
pip install streamlit streamlit-webrtc opencv-python mediapipe tensorflow numpy
```

## Run the demo

```bash
streamlit run main.py
```

Allow camera access in the browser when asked.

## Current status

- The demo draws hand landmarks and shows the mapped action, but `main.py` currently picks a gesture at random as a placeholder. Wiring in `hand_gesture_model.h5` for real predictions is the next step.
- Actions are displayed on screen; they are not yet sent to Spotify.

## Tech stack

TensorFlow / Keras, MediaPipe, OpenCV, Streamlit, streamlit-webrtc.
