import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import config

class ObjectDetector:
    def __init__(self, model_path=config.MODEL_PATH):
        """Initialize the model."""
        try:
            self.model = keras.models.load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def preprocess_frame(self, frame):
        """Resize and preprocess frame for the model."""
        frame_resized = cv2.resize(frame, config.INPUT_SHAPE)
        frame_preprocessed = preprocess_input(frame_resized)
        return frame_preprocessed

    def predict(self, frame_preprocessed):
        """Run prediction on a preprocessed frame."""
        predictions = self.model.predict(np.array([frame_preprocessed]))
        return predictions

    def process_frame(self, frame):
        """
        Takes a raw frame, runs prediction, and draws overlays.
        Returns the processed frame.
        """
        # 1. Prediction logic
        frame_preprocessed = self.preprocess_frame(frame)
        predictions = self.predict(frame_preprocessed)
        
        pred_index = np.argmax(predictions)
        pred_name = config.CATEGORIES[pred_index]
        pred_prob = round(predictions[0][pred_index] * 100, 2)

        # 2. Image Processing (Contours)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find largest contour
        max_area = 0
        max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

        # 3. drawing Logic
        if max_contour is not None:
            x, y, w, h = cv2.boundingRect(max_contour)
            xmin, ymin, xmax, ymax = x, y, x + w, y + h
            
            # Choose color based on detection
            if pred_name in config.CATEGORIES:
                # Logic from original: Red if detected?, Original had weird logic
                # Original: if pred_name in categories: color = RED else GREEN
                # Actually original was: if pred_name in categories: RED else GREEN. 
                # Wait, categories list contains ALL potential classes. So it's always RED?
                # Let's keep original behavior:
                box_color = config.COLOR_RED
            else:
                box_color = config.COLOR_GREEN
                
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_color, thickness=2)

            # Draw Label
            text = f"{pred_name} ({pred_prob}%)"
            cv2.putText(frame, text, (xmin+5, ymin+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, config.TEXT_COLOR, 2)

            # Draw Top 3
            top3_indices = np.argsort(predictions[0])[::-1][:3]
            for i, idx in enumerate(top3_indices):
                if config.CATEGORIES[idx] == pred_name:
                    continue
                # Offset usually starts below
                info_text = f"{config.CATEGORIES[idx]}: {round(predictions[0][idx]*100,2)}%"
                cv2.putText(frame, info_text, (10, 20*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.TEXT_COLOR, 1)

        return frame
