import cv2
import numpy as np
from ultralytics import YOLOWorld
import config

class ObjectDetector:
    def __init__(self, model_path="yolov8s-world.pt"):
        """Initialize the YOLO-World model."""
        self.model = None
        self.error_msg = None
        try:
            # Load YOLO-World model (will download on first run)
            # using 'yolov8s-world.pt' (Small) for balance of speed/accuracy
            self.model = YOLOWorld(model_path)
            
            # Define custom classes (English prompts)
            self.class_names = [
                'custard apple', 'apple', 'banana', 'lemon', 'mango', 
                'orange', 'tomato', 'person', 'pen', 'smart phone'
            ]
            
            # Set classes in the model
            self.model.set_classes(self.class_names)
            
            print(f"YOLO-World Model loaded successfully from {model_path}")
            
        except Exception as e:
            print(f"Error loading YOLO-World model: {e}")
            self.error_msg = str(e)
            self.model = None

    def process_frame(self, frame, conf_threshold=0.2, iou_threshold=0.5):
        """
        Takes a raw frame (BGR), runs prediction, and draws overlays.
        Returns the processed frame (BGR).
        """
        if self.model is None:
            msg = f"Err: {self.error_msg}" if self.error_msg else "Model Error"
            # Split message if too long
            y0, dy = 50, 30
            for i, line in enumerate(msg.split(' ')):
                 # Simple multiline
                 y = y0 + i*dy
                 if y > 300: break
                 cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return frame

        # Run inference
        # stream=True for video sources/efficiency, but here we do single image effectively
        results = self.model.predict(frame, conf=conf_threshold, iou=iou_threshold)

        # Draw results
        # Ultralytics results[0].plot() returns the BGR numpy array with boxes drawn
        annotated_frame = results[0].plot()

        return annotated_frame
