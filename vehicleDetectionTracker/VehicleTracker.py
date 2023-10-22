import time
import cv2
import numpy as np
import base64
from collections import defaultdict
from ultralytics import YOLO

class VehicleTracker:
    """
        Initialize the VehicleTracker.

        Args:
            model_path (str): Path to the YOLO model file.
            speed_threshold_min_kmph (float): Minimum speed threshold in km/h for vehicle tracking.
            speed_threshold_max_kmph (float): Maximum speed threshold in km/h for vehicle tracking.
    """
    def __init__(self, model_path="yolov8n.pt", speed_threshold_min_kmph=20.0, speed_threshold_max_kmph=120.0):
        self.model = YOLO(model_path)
        self.speed_threshold_min_kmph = speed_threshold_min_kmph
        self.speed_threshold_max_kmph = speed_threshold_max_kmph
        self.frame_rate = 30  # Frames per second (fps)
        self.pixel_scale = 0.1  # 1 pixel equals 0.1 meters
        self.track_history = defaultdict(lambda: [])
        self.frame_counter = 0

    def assign_color(self, avg_color):
        # Define ranges for common vehicle colors in RGB format
        color_ranges = {
            "Red": (np.array([120, 0, 0]), np.array([255, 60, 60])),
            "Blue": (np.array([0, 0, 120]), np.array([60, 60, 255])),
            "Green": (np.array([0, 120, 0]), np.array([60, 255, 60])),
            "Yellow": (np.array([180, 180, 0]), np.array([255, 255, 60])),
            "White": (np.array([200, 200, 200]), np.array([255, 255, 255])),
            "Black": (np.array([0, 0, 0]), np.array([50, 50, 50])),
            "Silver": (np.array([160, 160, 160]), np.array([220, 220, 220])),
            "Gray": (np.array([80, 80, 80]), np.array([160, 160, 160])),
            "Orange": (np.array([220, 120, 0]), np.array([255, 180, 40])),
            "Purple": (np.array([80, 0, 80]), np.array([160, 60, 160])),
        }

        # Check the average color against defined color ranges
        for color, (lower_bound, upper_bound) in color_ranges.items():
            if np.all(avg_color >= lower_bound) and np.all(avg_color <= upper_bound):
                return color

        # If the color doesn't match any defined ranges, return "Unknown"
        return "Unknown"

    def process_frame(self, frame):
        """
        Process a frame to detect and track vehicles.

        Args:
            frame (numpy.ndarray): Input frame for processing.

        Returns:
            dict or None: Processed information including tracked vehicles' details and the annotated frame in base64.
        """
        results = self.model.track(frame, persist=True)
        processed_info = []

        # Check if there are detected objects
        if results[0].boxes is not None:
            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            # Check if there are IDs associated with the detections
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
            else:
                track_ids = []  # No IDs available
                
            annotated_frame = results[0].plot()

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = self.track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                # Initialize speed_kmph with a default value
                speed_kmph = 0.0
                direction = None  # Initialize direction as None
                color_label = None
                roi_base64 = None # Base64-encoded ROI image
                timestamp = int(time.time())

                if len(track) >= 2:
                    x1, y1 = track[0]  # Initial position
                    x2, y2 = track[-1]  # Current position
                        
                    # Calculate speed
                    distance_pixels = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    speed_pixels_per_frame = distance_pixels / len(track)
                    speed_kmph = speed_pixels_per_frame * self.frame_rate * self.pixel_scale * 3.6

                    # Calculate direction based on position difference
                    dx = x2 - x1
                    dy = y2 - y1

                    if abs(dx) > abs(dy):
                        if dx > 0:
                            direction = "Right"
                        else:
                            direction = "Left"
                    else:
                        if dy > 0:
                            direction = "Down"
                        else:
                            direction = "Up"

                    # Check if speed_kmph is within the specified threshold
                    if self.speed_threshold_min_kmph <= speed_kmph <= self.speed_threshold_max_kmph:
                        # Draw speed information on the frame
                        text = f"Vehicle {track_id}: Speed {speed_kmph:.2f} km/h"
                        cv2.putText(annotated_frame, text, (int(x), int(y) - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

                    # Color estimation
                    roi = frame[int(y):int(y + h), int(x):int(x + w)]  # Define the ROI
                    if roi.size > 0:
                        # Extract color samples (e.g., from the top of the ROI)
                        color_samples = roi[0:10, :, :]
                        # Calculate the average color
                        avg_color = np.mean(color_samples, axis=(0, 1))
                        # Assign a color label based on the average color
                        color_label = self.assign_color(avg_color)
                        # Crop the ROI and convert it to base64
                        _, buffer = cv2.imencode('.jpg', roi)
                        roi_base64 = base64.b64encode(buffer).decode('utf-8')

                    if len(track) > 30:  # retain 90 tracks for 90 frames
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(
                        annotated_frame,
                        [points],
                        isClosed=False,
                        color=(230, 230, 230),
                        thickness=10,
                    )

                    processed_info.append({
                        'vehicle_id': track_id,
                        'direction': direction,
                        'color': color_label,
                        'timestamp': timestamp,
                        'speed_kmph': speed_kmph if self.speed_threshold_min_kmph <= speed_kmph <= self.speed_threshold_max_kmph else None,
                        'roi_base64': roi_base64
                    })

                # Convert annotated frame to base64
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                frame_base64 = base64.b64encode(buffer).decode()

                return {
                    'processed_info': processed_info,
                    'frame_base64': frame_base64
                }

            else:
                return None