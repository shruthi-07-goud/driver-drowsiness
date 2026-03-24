import streamlit as st
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.title("🚗 Driver Drowsiness Detection (Live Website)")

# Load cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.eye_closed_frames = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        status = "SAFE"

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)

            if len(eyes) == 0:
                self.eye_closed_frames += 1
            else:
                self.eye_closed_frames = 0

        if self.eye_closed_frames > 10:
            status = "DROWSY"
            cv2.putText(img, "WAKE UP!", (50,100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        cv2.putText(img, f"Status: {status}", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        return img


webrtc_streamer(key="drowsiness", video_processor_factory=VideoProcessor)
