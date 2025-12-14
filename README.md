# Computer Vision Fruit Classifier
A real-time Flask application and local script for classifying fruits and objects using a MobileNetV2 model with OpenCV.

## Features
- **Real-time Detection**: Uses webcam input to classify objects.
- **Classification**: Detects specific fruits/objects (Apple, Banana, Lemon, etc.).
- **Visual Feedback**: Draws bounding boxes (Green/Red) and labels with confidence scores.
- **Web Interface**: View the stream via a web browser.
- **Local Interface**: Run as a standalone desktop script.

## Project Structure
- `app.py`: Main Flask application for web deployment.
- `local_run.py`: Script to run object detection locally with a window.
- `core.py`: Contains the `ObjectDetector` class with all ML and image processing logic.
- `config.py`: Configuration for model paths, categories, and colors.
- `requirements.txt`: Python dependencies.
- `Procfile`: Configuration for Heroku/Render deployment.

## Installation

1. **Clone the repository** (or download the files).
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Ensure you have Python installed.*

3. **Model File**:
   Ensure `mobilenet.h5` is in the root directory.

## Usage

### 1. Web Application
Run the Flask server:
```bash
python app.py
```
Open your browser to `http://127.0.0.1:5000`.

### 2. Local Desktop Run
Run the standalone script:
```bash
python local_run.py
```
A window will open showing the camera feed with detection. Press **'q'** to quit.

## Deployment
This project includes a `Procfile` and `requirements.txt` suitable for deployment on platforms like Heroku or Render.

**Note on Cloud Deployment**:
The current implementation of `cv2.VideoCapture(0)` attempts to access the server's physical webcam. On cloud instances without a camera, the video stream will typically fail or show errors. For true web-based client-side camera streaming, integration with WebRTC or JavaScript-based capture is recommended.
