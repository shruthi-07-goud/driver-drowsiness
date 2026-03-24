from flask import Flask, render_template, Response
import cv2
import winsound
import time
import pyttsx3

app = Flask(__name__)

# Voice engine
engine = pyttsx3.init()

# Load cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

camera = cv2.VideoCapture(0)

running = True  # for start/stop

def speak(text):
    engine.say(text)
    engine.runAndWait()

def generate_frames():
    global running
    eye_closed_frames = 0
    blink_count = 0
    last_eye_open = True
    start_time = time.time()

    while True:
        if not running:
            continue

        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        status = "SAFE"
        current_time = int(time.time() - start_time)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # 👀 Attention detection (face center)
            frame_center = frame.shape[1] // 2
            face_center = x + w // 2

            if abs(face_center - frame_center) > 100:
                cv2.putText(frame, "LOOK AT ROAD!", (50, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # 👀 Eye detection
            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            if len(eyes) == 0:
                eye_closed_frames += 1
                if last_eye_open:
                    blink_count += 1
                last_eye_open = False
            else:
                eye_closed_frames = 0
                last_eye_open = True

            # 😮 Yawning detection
            roi_gray_lower = roi_gray[int(h/2):h, :]
            mouths = mouth_cascade.detectMultiScale(
                roi_gray_lower,
                scaleFactor=1.5,
                minNeighbors=15
            )

            if len(mouths) > 0:
                cv2.putText(frame, "YAWNING!", (50,150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)

                winsound.Beep(2000, 300)
                speak("You are yawning")

        # 🚨 Drowsiness logic
        if eye_closed_frames > 5:
            status = "WARNING"

        if eye_closed_frames > 10:
            status = "DROWSY"

            for _ in range(2):
                winsound.Beep(800, 300)

            speak("Wake up driver")

            # 📸 Screenshot
            cv2.imwrite(f"sleep_{int(time.time())}.jpg", frame)

        # 📊 Drowsiness %
        drowsy_percent = min(100, eye_closed_frames * 5)

        # ☕ Break message
        if current_time > 20 and status == "DROWSY":
            cv2.putText(frame, "TAKE A BREAK!", (50,200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        # Display info
        cv2.putText(frame, f"Status: {status}", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

        cv2.putText(frame, f"Blinks: {blink_count}", (50,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

        cv2.putText(frame, f"Time: {current_time}s", (50,250),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.putText(frame, f"Drowsiness: {drowsy_percent}%", (50,300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start')
def start():
    global running
    running = True
    return "Started"


@app.route('/stop')
def stop():
    global running
    running = False
    return "Stopped"


if __name__ == "__main__":
    app.run(debug=True)
