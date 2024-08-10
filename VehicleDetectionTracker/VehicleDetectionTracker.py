import json
import math
import cv2
import base64
from collections import defaultdict
from ultralytics import YOLO
import numpy as np
from ultralytics.utils.plotting import colors
from VehicleDetectionTracker.color_classifier.classifier import Classifier as ColorClassifier
from VehicleDetectionTracker.model_classifier.classifier import Classifier as ModelClassifier
from datetime import datetime

class VehicleDetectionTracker:

    def __init__(self, model_path="yolov8n.pt"):
        """
        Initialize the VehicleDetection class.

        Args:
            model_path (str): Path to the YOLO model file.
        """
        # Load the YOLO model and set up data structures for tracking.
        self.model = YOLO(model_path)
        self.track_history = defaultdict(lambda: [])  # History of vehicle tracking
        self.detected_vehicles = set()  # Set of detected vehicles
        self.color_classifier = None
        self.model_classifier = None
        self.vehicle_timestamps = defaultdict(list)  # Keep track of timestamps for each tracked vehicle


    def _initialize_classifiers(self):
        if self.color_classifier is None:
            self.color_classifier = ColorClassifier()
        if self.model_classifier is None:
            self.model_classifier = ModelClassifier()

    def _map_direction_to_label(self, direction):
        # Define direction ranges in radians and their corresponding labels
        direction_ranges = {
            (-math.pi / 8, math.pi / 8): "Right",
            (math.pi / 8, 3 * math.pi / 8): "Bottom Right",
            (3 * math.pi / 8, 5 * math.pi / 8): "Bottom",
            (5 * math.pi / 8, 7 * math.pi / 8): "Bottom Left",
            (7 * math.pi / 8, -7 * math.pi / 8): "Left",
            (-7 * math.pi / 8, -5 * math.pi / 8): "Top Left",
            (-5 * math.pi / 8, -3 * math.pi / 8): "Top",
            (-3 * math.pi / 8, -math.pi / 8): "Top Right"
        }
        for angle_range, label in direction_ranges.items():
            if angle_range[0] <= direction <= angle_range[1]:
                return label
        return "Unknown"  # Return "Unknown" if the direction doesn't match any defined range


    def _encode_image_base64(self, image):
        """
        Encode an image as base64.

        Args:
            image (numpy.ndarray): The image to be encoded.

        Returns:
            str: Base64-encoded image.
        """
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode()
        return image_base64
    
    def _decode_image_base64(self, image_base64):
        """
        Decode a base64-encoded image.

        Args:
            image_base64 (str): Base64-encoded image data.

        Returns:
            numpy.ndarray or None: Decoded image as a numpy array or None if decoding fails.
        """
        try:
            image_data = base64.b64decode(image_base64)
            image_np = np.frombuffer(image_data, dtype=np.uint8)
            image = cv2.imdecode(image_np, flags=cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            return None
        
    def _increase_brightness(self, image, factor=1.5):
        """
        Increases the brightness of an image by multiplying its pixels by a factor.

        :param image: The input image in numpy array format.
        :param factor: The brightness increase factor. A value greater than 1 will increase brightness.
        :return: The image with increased brightness.
        """
        brightened_image = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        return brightened_image

    def _convert_meters_per_second_to_kmph(self, meters_per_second):
        # 1 m/s is approximately 3.6 km/h
        kmph = meters_per_second * 3.6
        return kmph

    def process_frame_base64(self, frame_base64, frame_timestamp):
        """
        Process a base64-encoded frame to detect and track vehicles.

        Args:
            frame_base64 (str): Base64-encoded input frame for processing.

        Returns:
            dict or None: Processed information including tracked vehicles' details and the annotated frame in base64,
            or an error message if decoding fails.
        """
        frame = self._decode_image_base64(frame_base64)
        if frame is not None:
            return self.process_frame(frame, frame_timestamp)
        else:
            return {
                "error": "Failed to decode the base64 image"
            }

    def process_frame(self, frame, frame_timestamp):
        """
        Process a single video frame to detect and track vehicles.

        Args:
            frame (numpy.ndarray): Input frame for processing.

        Returns:
            dict: Processed information including tracked vehicles' details, the annotated frame in base64, and the original frame in base64.
        """
        self._initialize_classifiers()
        response = {
            "number_of_vehicles_detected": 0,  # Counter for vehicles detected in this frame
            "detected_vehicles": [],  # List of information about detected vehicles
            "annotated_frame_base64": None,  # Annotated frame as a base64 encoded image
            "original_frame_base64": None  # Original frame as a base64 encoded image
        }
        # Process a single video frame and return detection results, an annotated frame, and the original frame as base64.
        results = self.model.track(self._increase_brightness(frame), persist=True, tracker="bytetrack.yaml")  # Perform vehicle tracking in the frame
        if results is not None and results[0] is not None and results[0].boxes is not None and results[0].boxes.id is not None:
            # Obtain bounding boxes (xywh format) of detected objects
            boxes = results[0].boxes.xywh.cpu()
            # Extract confidence scores for each detected object
            conf_list = results[0].boxes.conf.cpu()
            # Get unique IDs assigned to each tracked object
            track_ids = results[0].boxes.id.int().cpu().tolist()
            # Obtain the class labels (e.g., 'car', 'truck') for detected objects
            clss = results[0].boxes.cls.cpu().tolist()
            # Retrieve the names of the detected objects based on class labels
            names = results[0].names
            # Get the annotated frame using results[0].plot() and encode it as base64
            annotated_frame = results[0].plot()

            for box, track_id, cls, conf in zip(boxes, track_ids, clss, conf_list):
                x, y, w, h = box
                label = str(names[cls])
                # Bounding box plot
                bbox_color = colors(cls, True)
                track_thickness=2
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                # Retrieve or create a list to store the tracking history of the current vehicle (identified by track_id).
                track = self.track_history[track_id]
                # Append the current position (x, y) to the tracking history list.
                track.append((float(x), float(y)))
                # Limit the tracking history to the last 30 positions to avoid excessive memory usage.
                max_history_length = 30
                if len(track) > max_history_length:
                    track.pop(0)
                # Combine the tracked points into a NumPy array for drawing a polyline.
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                # Draw a polyline (tracking lines) on the annotated frame using the combined points.
                cv2.polylines(annotated_frame, [points], isClosed=False, color=bbox_color, thickness=track_thickness)

                if track_id not in self.vehicle_timestamps:
                    self.vehicle_timestamps[track_id] = {"timestamps": [], "positions": []}  # Initialize timestamps and positions lists

                # Store the timestamp for this frame
                self.vehicle_timestamps[track_id]["timestamps"].append(frame_timestamp)
                self.vehicle_timestamps[track_id]["positions"].append((x, y))
                # Calculate the speed if there are enough timestamps (at least 2)
                timestamps = self.vehicle_timestamps[track_id]["timestamps"]
                positions = self.vehicle_timestamps[track_id]["positions"]
                speed_kph = None
                reliability = 0.0
                direction_label = None
                direction = None
                if len(timestamps) >= 2:
                    delta_t_list = []
                    distance_list = []
                    # Calculate time intervals (delta_t) and distances traveled between successive frames
                    for i in range(1, len(timestamps)):
                        t1, t2 = timestamps[i - 1], timestamps[i]
                        delta_t = t2.timestamp() - t1.timestamp()
                        if delta_t > 0:
                            x1, y1 = positions[i - 1]
                            x2, y2 = positions[i]
                            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                            delta_t_list.append(delta_t)
                            distance_list.append(distance)

                    
                    # Calculate speeds in meters per second (mps) for each frame and then average them
                    speeds = [distance / delta_t for distance, delta_t in zip(distance_list, delta_t_list)]
                    if len(speeds) > 0:
                        avg_speed_mps = sum(speeds) / len(speeds)
                    else:
                        avg_speed_mps = None

                    # Convert the average speed from meters per second (mps) to kilometers per hour (kph)
                    if avg_speed_mps is not None:
                        speed_kph = self._convert_meters_per_second_to_kmph(avg_speed_mps)
                    else:
                        speed_kph = None
                    # Calculate the direction based on the change in position between the first and last frame
                    initial_x, initial_y = positions[0]
                    final_x, final_y = positions[-1]
                    direction = math.atan2(final_y - initial_y, final_x - initial_x)
                    direction_label = self._map_direction_to_label(direction)

                    # Calculate reliability based on the number of samples used
                    if len(timestamps) < 5:
                        reliability = 0.5  # Low reliability if there are less than 5 samples
                    elif len(timestamps) < 10:
                        reliability = 0.7  # Moderate reliability if there are between 5 and 10 samples
                    else:
                        reliability = 1.0  # High reliability if there are 10 or more samples


                # If the vehicle is new, process it
                self.detected_vehicles.add(track_id)  # Add the vehicle to the set of detected vehicles
                response["number_of_vehicles_detected"] += 1  # Increment the counter

                # Extract the frame of the detected vehicle
                vehicle_frame = frame[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]
                vehicle_frame_base64 = self._encode_image_base64(vehicle_frame)
                color_info = self.color_classifier.predict(vehicle_frame)
                color_info_json = json.dumps(color_info)
                model_info = self.model_classifier.predict(vehicle_frame)
                model_info_json = json.dumps(model_info)
    
                 # Add vehicle information to the response
                response["detected_vehicles"].append({
                    "vehicle_id": track_id,
                    "vehicle_type": label,
                    "detection_confidence": conf.item(),
                    "vehicle_coordinates": {
                        "x": x.item(),
                        "y": y.item(), 
                        "width": w.item(), 
                        "height": h.item()
                    },
                    "vehicle_frame_base64": vehicle_frame_base64,
                    "vehicle_frame_timestamp": frame_timestamp, 
                    "color_info": color_info_json,
                    "model_info": model_info_json,
                    "speed_info": {
                        "kph": speed_kph, 
                        "reliability": reliability,
                        "direction_label": direction_label,
                        "direction": direction
                    }
                })
                    
            annotated_frame_base64 = self._encode_image_base64(annotated_frame)
            response["annotated_frame_base64"] = annotated_frame_base64

        # Encode the original frame as base64
        original_frame_base64 = self._encode_image_base64(frame)
        response["original_frame_base64"] = original_frame_base64

        return response

    def process_video(self, video_path, result_callback):
        """
        Process a video by calling a callback for each frame's results.

        Args:
            video_path (str): Path to the video file.
            result_callback (function): A callback function to handle the processing results for each frame.
        """
        # Process a video frame by frame, calling a callback with the results.
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            success, frame = cap.read()
            if success:
                frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
                print(f"Frame rate: {frame_rate} FPS")
                timestamp = datetime.now()
                response = self.process_frame(frame, timestamp)
                if 'annotated_frame_base64' in response:
                    annotated_frame = self._decode_image_base64(response['annotated_frame_base64'])
                    if annotated_frame is not None:
                        # Display the annotated frame in a window
                        cv2.imshow("Video Detection Tracker - YOLOv8 + bytetrack", annotated_frame)
                # Call the callback with the response
                result_callback(response)
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Break the loop if the end of the video is reached
                break

        # Release the video capture object and close the display window
        cap.release()
        cv2.destroyAllWindows()
