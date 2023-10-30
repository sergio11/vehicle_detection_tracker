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
        self.color_classifier = ColorClassifier()
        self.model_classifier = ModelClassifier()
        self.vehicle_timestamps = defaultdict(list)  # Keep track of timestamps for each tracked vehicle

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
    
    def _calculate_scale(self, fov, frame):
        # These values can be obtained dynamically from the frame
        image_width = frame.shape[1]  # Width of the frame
        image_height = frame.shape[0]  # Height of the frame
        # FOV is given in degrees, we convert it to radians.
        fov_rad = math.radians(fov)
        
        print(f'Image width: {image_width} pixels')
        print(f'Image height: {image_height} pixels')
        # Calculate the visible width and height at the focal distance.
        focal_distance = (image_width / 2) / math.tan(fov_rad / 2)
        visible_width = 2 * focal_distance * math.tan(fov_rad / 2)
        visible_height = 2 * focal_distance * math.tan(fov_rad / 2)
        
        # Calculate the scale in meters per pixel.
        scale_x = visible_width / image_width
        scale_y = visible_height / image_height
        
        return scale_x, scale_y

    def _calculate_fov(self, sensor_size_mm, focal_length_mm):
        return 2 * math.degrees(math.atan(sensor_size_mm / (2 * focal_length_mm)))

    def _convert_pixels_to_meters(self, pixels_per_second, scale_x, scale_y):
        meters_per_second_x = pixels_per_second * scale_x
        meters_per_second_y = pixels_per_second * scale_y
        return meters_per_second_x, meters_per_second_y

    def _convert_meters_per_second_to_kmph(self, meters_per_second):
        # 1 m/s is approximately 3.6 km/h
        kmph = math.sqrt(meters_per_second[0] ** 2 + meters_per_second[1] ** 2) * 3.6
        return kmph

    def process_frame_base64(self, frame_base64, focal_length_mm, sensor_size_mm, frame_timestamp):
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
            return self.process_frame(frame, focal_length_mm, sensor_size_mm, frame_timestamp)
        else:
            return {
                "error": "Failed to decode the base64 image"
            }

    def process_frame(self, frame, focal_length_mm, sensor_size_mm, frame_timestamp):
        """
        Process a single video frame to detect and track vehicles.

        Args:
            frame (numpy.ndarray): Input frame for processing.

        Returns:
            dict: Processed information including tracked vehicles' details, the annotated frame in base64, and the original frame in base64.
        """
        response = {
            "number_of_vehicles_detected": 0,  # Counter for vehicles detected in this frame
            "detected_vehicles": [],  # List of information about detected vehicles
            "annotated_frame_base64": None,  # Annotated frame as a base64 encoded image
            "original_frame_base64": None  # Original frame as a base64 encoded image
        }
        # Process a single video frame and return detection results, an annotated frame, and the original frame as base64.
        results = self.model.track(self._increase_brightness(frame), persist=True, tracker="bytetrack.yaml")  # Perform vehicle tracking in the frame
        if results is not None and results[0] is not None and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            conf_list = results[0].boxes.conf.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            names = results[0].names
            # Get the annotated frame using results[0].plot() and encode it as base64
            annotated_frame = results[0].plot()
            fov_degress = self._calculate_fov(sensor_size_mm, focal_length_mm)

            for box, track_id, cls, conf in zip(boxes, track_ids, clss, conf_list):
                x, y, w, h = box
                label = str(names[cls])
                # Bounding box plot
                bbox_color = colors(cls, True)
                track_thickness=2
                # Tracking Lines plot
                track = self.track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 30:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
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
                if len(timestamps) >= 2:
                    t1, t2 = timestamps[-2], timestamps[-1]  # Get the last two timestamps
                    delta_t = (t2 - t1)  # Time elapsed in seconds
                    delta_t_seconds = delta_t.total_seconds()
                    if delta_t_seconds > 0:
                        # Calculate distance traveled between the two frames
                        x1, y1 = positions[-2]
                        x2, y2 = positions[-1]
                        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        # Calculate speed in pixels per second
                        speed_pxs_per_sec = distance / delta_t_seconds
                        scale_x, scale_y = self._calculate_scale(fov_degress, frame)
                        print(f'Scale in meters per pixel: ({scale_x}, {scale_y})')
                        speed_ms = self._convert_pixels_to_meters(speed_pxs_per_sec, scale_x, scale_y)
                        speed_kph = self._convert_meters_per_second_to_kmph(speed_ms)

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
                    "vehicle_coordinates": {"x": x.item(), "y": y.item(), "width": w.item(), "height": h.item()},
                    "vehicle_frame_base64": vehicle_frame_base64,
                    "color_info": color_info_json,
                    "model_info": model_info_json,
                    "speed_kph": speed_kph
                })
                    
            annotated_frame_base64 = self._encode_image_base64(annotated_frame)
            response["annotated_frame_base64"] = annotated_frame_base64

        # Encode the original frame as base64
        original_frame_base64 = self._encode_image_base64(frame)
        response["original_frame_base64"] = original_frame_base64

        return response

    def process_video(self, video_path, result_callback, focal_length_mm, sensor_size_mm):
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
                response = self.process_frame(frame, focal_length_mm, sensor_size_mm, timestamp)
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
