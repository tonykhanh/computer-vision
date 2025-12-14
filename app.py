from flask import Flask, render_template, Response
import cv2
from time import sleep
from core import ObjectDetector

# Initialize App & Model
app = Flask(__name__)

# Initialize Detector (Global load to avoid reloading per request if multiple workers - usually handled by gunicorn workers)
# Note: In production with multiple workers, this loads per worker.
try:
    detector = ObjectDetector()
except Exception as e:
    print("WARNING: Model could not be loaded. Ensure 'mobilenet.h5' is present.")
    detector = None

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/xinchao')
def XinChaoMoiNguoi():
    return "<h3>Xin chào mọi người</h3>"

def gen():
    """Video streaming generator function."""
    cap = cv2.VideoCapture(0)
    sleep(2) # Warmup

    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if detector:
            # Process frame using our core logic
            frame = detector.process_frame(frame)

        # Encode frame for web
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b"\r\n")

    cap.release()

@app.route('/mo_webcam')
def VideoTheoThuMuc():
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    app.run(debug=True)
