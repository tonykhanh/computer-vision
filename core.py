import cv2
import numpy as np
import config

try:
    # Try using lightweight tflite_runtime (for deployment)
    import tflite_runtime.interpreter as tflite
except ImportError:
    # Fallback to full tensorflow (for local development)
    try:
        import tensorflow.lite as tflite
    except ImportError:
        print("Error: Neither tflite_runtime nor tensorflow is installed.")
        tflite = None

class ObjectDetector:
    def __init__(self, model_path=config.MODEL_PATH):
        """Initialize the TFLite model."""
        self.interpreter = None
        if tflite:
            try:
                self.interpreter = tflite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                
                print(f"TFLite Model loaded successfully from {model_path}")
            except Exception as e:
                print(f"Error loading TFLite model: {e}")
                # Don't raise, allowing app to start with "Model Not Loaded" warning
        
    def preprocess_input(self, x):
        """
        Manual implementation of MobileNetV2 preprocess_input.
        Scales values from [0, 255] to [-1, 1].
        """
        x = x.astype(np.float32)
        x /= 127.5
        x -= 1.
        return x

    def preprocess_frame(self, frame):
        """Resize and preprocess frame for the model."""
        # Resize to (128, 128)
        frame_resized = cv2.resize(frame, config.INPUT_SHAPE)
        
        # Expand dims to (1, 128, 128, 3)
        frame_batch = np.expand_dims(frame_resized, axis=0)
        
        # Preprocess
        frame_preprocessed = self.preprocess_input(frame_batch)
        return frame_preprocessed

    def predict(self, frame_preprocessed):
        """Run prediction using TFLite interpreter."""
        if not self.interpreter:
            return np.zeros((1, len(config.CATEGORIES)))

        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], frame_preprocessed)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output tensor
        predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
        return predictions

    def process_frame(self, frame):
        """
        Takes a raw frame, runs prediction, and draws overlays.
        Returns the processed frame.
        """
        if not self.interpreter:
             cv2.putText(frame, "Model Not Loaded", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
             return frame

        # 1. Prediction logic
        frame_preprocessed = self.preprocess_frame(frame)
        predictions = self.predict(frame_preprocessed)
        
        pred_index = np.argmax(predictions)
        # Check boundary
        if pred_index < len(config.CATEGORIES):
            pred_name = config.CATEGORIES[pred_index]
        else:
            pred_name = "Unknown"
            
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
            
            box_color = config.COLOR_RED # Default color
                
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_color, thickness=2)

            # Draw Label
            text = f"{pred_name} ({pred_prob}%)"
            cv2.putText(frame, text, (xmin+5, ymin+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, config.TEXT_COLOR, 2)

            # Draw Top 3
            top3_indices = np.argsort(predictions[0])[::-1][:3]
            for i, idx in enumerate(top3_indices):
                if idx >= len(config.CATEGORIES): continue
                if config.CATEGORIES[idx] == pred_name:
                    continue
                # Offset usually starts below
                info_text = f"{config.CATEGORIES[idx]}: {round(predictions[0][idx]*100,2)}%"
                cv2.putText(frame, info_text, (10, 20*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.TEXT_COLOR, 1)

        return frame
