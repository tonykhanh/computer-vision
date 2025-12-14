from flask import Flask, render_template, Response, request
import cv2
import numpy as np
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

@app.route('/process_frame', methods=['POST'])
def process_frame_route():
    """
    API endpoint for processing frames sent from the client.
    Essential for serverless environments (Vercel) where server-side webcam is unavailable.
    """
    if 'image' not in request.files:
        return "No image provided", 400
    
    file = request.files['image']
    
    # Convert string data to numpy array
    npimg = np.frombuffer(file.read(), np.uint8)
    
    # Convert numpy array to image
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if detector:
        # Process frame using our core logic
        frame = detector.process_frame(frame)
    else:
        # Fallback if model failed to load
        cv2.putText(frame, "Model Not Loaded", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Encode frame for response
    ret, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()

    return Response(frame_bytes, mimetype='image/jpeg')

if __name__ == '__main__':
    # Port 5000 is often taken by AirPlay on MacOS (ControlCenter)
    app.run(debug=True, port=5000)
