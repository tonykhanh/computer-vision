import cv2
import numpy as np
from ultralytics import YOLOWorld
import config

class ObjectDetector:
    def __init__(self, model_path="yolov8s-world.pt"):
        """Initialize the YOLO-World model."""
        self.model = None
        try:
            # Load YOLO-World model (will download on first run)
            # using 'yolov8s-world.pt' (Small) for balance of speed/accuracy
            self.model = YOLOWorld(model_path)
            
            # Define custom classes (English prompts)
            # Mapping config.CATEGORIES (which might be Vietnamese or IDs) to English prompts
            # config.CATEGORIES: ['annona', 'apples', 'bananas', 'lemons', 'mango', 'oranges', 'tomatoes', 'human', 'pen', 'phone']
            
            # Create a mapping or just set the classes directly.
            # We want to detect THESE specific things.
            # YOLO-World uses text prompts.
            self.class_names = [
                'custard apple', # annona
                'apple',         # apples
                'banana',        # bananas
                'lemon',         # lemons
                'mango',         # mango
                'orange',        # oranges
                'tomato',        # tomatoes
                'person',        # human
                'pen',           # pen
                'smart phone'    # phone
            ]
            
            # Set classes in the model
            self.model.set_classes(self.class_names)
            
            print(f"YOLO-World Model loaded successfully from {model_path}")
            print(f"Active classes: {self.class_names}")
            
        except Exception as e:
            print(f"Error loading YOLO-World model: {e}")
            self.model = None

    def process_frame(self, frame, conf_threshold=0.2, iou_threshold=0.5):
        """
        Takes a raw frame (BGR), runs prediction, and draws overlays.
        Returns the processed frame (BGR).
        """
        if self.model is None:
            cv2.putText(frame, "Model Error", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame

        # Run inference
        # stream=True for video sources/efficiency, but here we do single image effectively
        results = self.model.predict(frame, conf=conf_threshold, iou=iou_threshold)

        # Draw results
        # Ultralytics results[0].plot() returns the BGR numpy array with boxes drawn
        annotated_frame = results[0].plot()

        return annotated_frame
