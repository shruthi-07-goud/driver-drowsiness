import streamlit as st
import cv2
import time
import pyttsx3

st.title("🚗 Driver Drowsiness Detection")

# Initialize voice
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Load cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

eye_closed_frames = 0
blink_count = 0
last_eye_open = True
start_time = time.time()

while run:
    ret, frame = camera.read()
    if not ret:
        st.write("Camera error")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    status = "SAFE"
    current_time = int(time.time() - start_time)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Eye detection
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)

        if len(eyes) == 0:
            eye_closed_frames += 1
            if last_eye_open:
                blink_count += 1
            last_eye_open = False
        else:
            eye_closed_frames = 0
            last_eye_open = True

        # Mouth detection (yawning)
        roi_gray_lower = roi_gray[int(h/2):h, :]
        mouths = mouth_cascade.detectMultiScale(roi_gray_lower, 1.5, 15)

        if len(mouths) > 0:
            cv2.putText(frame, "YAWNING!", (50,150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)
            speak("You are yawning")

    # Drowsiness logic
    if eye_closed_frames > 5:
        status = "WARNING"

    if eye_closed_frames > 10:
        status = "DROWSY"
        speak("Wake up driver")

    drowsy_percent = min(100, eye_closed_frames * 5)

    # Display text
    cv2.putText(frame, f"Status: {status}", (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

    cv2.putText(frame, f"Blinks: {blink_count}", (50,100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    cv2.putText(frame, f"Time: {current_time}s", (50,200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.putText(frame, f"Drowsiness: {drowsy_percent}%", (50,250),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    FRAME_WINDOW.image(frame, channels="BGR")

camera.release()
