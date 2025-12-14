import cv2
from time import sleep
from core import ObjectDetector

def main():
    print("Initializing Camera & Model...")
    detector = ObjectDetector()
    cap = cv2.VideoCapture(0)
    sleep(2)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Running. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        frame = detector.process_frame(frame)

        # Show result
        cv2.imshow("Object Detection - Local Run", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
