from VehicleDetectionTracker.VehicleTracker import VehicleTracker
import cv2

# Create a VehicleDetectionTracker instance
tracker = VehicleTracker()

# Process a frame
frame = cv2.imread('your_frame.jpg')
result = tracker.process_frame(frame)

# Access processed information
if result:
    processed_info = result['processed_info']
    annotated_frame_base64 = result['frame_base64']
    # Your custom processing logic here