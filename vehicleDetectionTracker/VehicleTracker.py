import time
import cv2
import numpy as np
import base64
from collections import defaultdict
from ultralytics import YOLO
import pytesseract


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
        self.prev_x_center = None
        self.prev_y_center = None

    
    def _convert_frame_to_base64(self, frame):
        """
        Convert the annotated frame to a base64-encoded string.

        Args:
            frame (numpy.ndarray): The annotated frame.

        Returns:
            str: Base64-encoded frame.
        """
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode()
    

    def _enhance_night_vision(self, frame):
        """
        Enhance the night vision of a frame by improving contrast, brightness, and reducing noise.

        Args:
            frame (numpy.ndarray): Input frame to enhance.

        Returns:
            numpy.ndarray: Enhanced frame or the original frame if enhancement is not possible.
        """
        try:
            # Ensure that the frame is in the correct format (BGR)
            if frame is None or len(frame.shape) != 3 or frame.shape[2] != 3:
                raise ValueError("Invalid frame format. Please ensure the input frame is in BGR format.")

            # Apply adaptive histogram equalization for contrast enhancement
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            enhanced_frame = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray_frame)

            # Increase brightness and add some contrast
            enhanced_frame = cv2.convertScaleAbs(enhanced_frame, alpha=1.2, beta=20)

            # Convert the enhanced grayscale frame back to color
            enhanced_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_GRAY2BGR)

            # Apply median blur for noise reduction
            enhanced_frame = cv2.medianBlur(enhanced_frame, 5)

            return enhanced_frame
        except Exception as e:
            # If an exception occurs during enhancement, return the original frame
            print(f"Error during night vision enhancement: {e}")
            return frame

    
    def process_frame_base64(self, frame_base64):
        """
        Process a frame to detect and track vehicles.

        Args:
            frame_base64 (str): Base64-encoded input frame for processing.

        Returns:
            dict or None: Processed information including tracked vehicles' details and the annotated frame in base64.
        """
        # Decode the base64-encoded frame
        frame = base64.b64decode(frame_base64)
        frame = np.frombuffer(frame, dtype=np.uint8)
        frame = cv2.imdecode(frame, flags=cv2.IMREAD_COLOR)
        return self.process_frame(frame)
    
    def _determine_vehicle_color(self, vehicle_image):
        # Resize the image for analysis
        resized_image = cv2.resize(vehicle_image, (10, 10))

        # Calculate the average color of the resized image
        avg_color = np.mean(resized_image, axis=(0, 1))

        # Define color ranges in RGB format
        color_ranges = {
            "Red": (np.array([150, 0, 0]), np.array([255, 50, 50])),
            "Blue": (np.array([0, 0, 150]), np.array([50, 50, 255])),
            "Green": (np.array([0, 150, 0]), np.array([50, 255, 50])),
            "Yellow": (np.array([200, 200, 0]), np.array([255, 255, 50])),
            "White": (np.array([200, 200, 200]), np.array([255, 255, 255])),
            "Black": (np.array([0, 0, 0]), np.array([50, 50, 50])),
            "Silver": (np.array([160, 160, 160]), np.array([220, 220, 220])),
            "Gray": (np.array([80, 80, 80]), np.array([160, 160, 160])),
            "Orange": (np.array([220, 120, 0]), np.array([255, 180, 40])),
            "Purple": (np.array([80, 0, 80]), np.array([160, 60, 160])),
            "Brown": (np.array([100, 40, 0]), np.array([150, 70, 20])),
            "Beige": (np.array([200, 180, 160]), np.array([255, 220, 190])),
            "Gold": (np.array([170, 130, 0]), np.array([220, 180, 40])),
            "Cyan": (np.array([0, 180, 180]), np.array([50, 230, 230])),
            "Magenta": (np.array([180, 0, 180]), np.array([230, 50, 230])),
            "Lime": (np.array([0, 180, 0]), np.array([50, 230, 50]))

        }

        # Compare the average color with color ranges
        for color, (lower_bound, upper_bound) in color_ranges.items():
            if np.all(avg_color >= lower_bound) and np.all(avg_color <= upper_bound):
                return color

        # If it doesn't match any defined color range, mark it as "Unknown"
        return "Unknown"


    def _get_license_plate(self, roi):
        """
        Get the license plate from the ROI using OCR.

        Args:
            roi (numpy.ndarray): Region of interest (ROI) image containing the license plate.

        Returns:
            str: Extracted license plate text. Returns None if no license plate is found.
        """
        try:
            if roi is None:
                return None

            # Convert the ROI to grayscale
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Apply a binary threshold using Otsu's method to highlight characters
            _, thresholded = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Optional: Apply opening and closing operations to enhance text quality
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
            thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

            # Use Tesseract OCR to extract text
            license_plate_text = pytesseract.image_to_string(thresholded)

            return license_plate_text.strip()
        except Exception as e:
            # Handle any exceptions that may occur during OCR
            print(f"An error occurred during license plate extraction: {str(e)}")
            return None
        

    def process_frame(self, frame):
        # Enhance the night vision of the frame
        enhanced_frame = self._enhance_night_vision(frame)

        # Execute the tracking using the YOLO model
        results = self.model.track(frame, persist=True, conf=0.3, iou=0.5, show=True, tracker="bytetrack.yaml")
        processed_info = []
        num_vehicles_detected = 0

        if results[0].boxes is not None:
            # Define the first ROI with a height of 30 pixels, lowered on the Y-axis
            frame_width, frame_height = frame.shape[1], frame.shape[0]
            roi_height = 30  # Height of both ROIs
            distance_between_rois = 40  # Vertical distance between ROIs
            
            center_y = frame_height // 2

            first_roi_top = center_y + (distance_between_rois // 2)
            first_roi = (0, first_roi_top, frame_width, first_roi_top + roi_height)

            second_roi_top = first_roi[3] + distance_between_rois
            second_roi = (0, second_roi_top, frame_width, second_roi_top + roi_height)

            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            num_vehicles_detected = len(boxes)
            annotated_frame = results[0].plot()
            # Padding around the vehicle
            vehicle_padding = 10

            for i, box in enumerate(boxes):
                x, y, w, h = box
                x_center = x + w / 2
                y_center = y + h / 2
                print(f"New box - x_center: {x_center}, y_center: {y_center}")

                timestamp = int(time.time())
                # Apply padding to the vehicle region
                x1 = max(0, int(x - vehicle_padding))
                y1 = max(0, int(y - vehicle_padding))
                x2 = min(frame.shape[1], int(x + w + vehicle_padding))
                y2 = min(frame.shape[0], int(y + h + vehicle_padding))

                vehicle_image = frame[y1:y2, x1:x2]
                color_label = self._determine_vehicle_color(vehicle_image)
                direction = None
                
                if self.prev_x_center is not None:
                    # Calculate direction based on position difference
                    dx = x_center - self.prev_x_center
                    dy = y_center - self.prev_y_center

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

                self.prev_x_center = x_center
                self.prev_y_center = y_center

                # Calculate the speed


                processed_info.append({
                    'direction': direction,
                    'color': color_label,
                    'timestamp': timestamp,
                    'license_plate': self._get_license_plate(vehicle_image),
                    'vehicle_image_base64': self._convert_frame_to_base64(vehicle_image),
                    #'speed_m_s': vehicle_speed_m_s,  # Speed in m/s
                    #'speed_kmph': vehicle_speed_kmph  # Speed in km/h
                })

                # Draw the first ROI
                cv2.rectangle(annotated_frame, (first_roi[0], first_roi[1]), (first_roi[2], first_roi[3]), (0, 255, 0), 2)
                cv2.putText(annotated_frame, "ROI1", (first_roi[0], first_roi[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # Draw the second ROI
                cv2.rectangle(annotated_frame, (second_roi[0], second_roi[1]), (second_roi[2], second_roi[3]), (0, 0, 255), 2)
                cv2.putText(annotated_frame, "ROI2", (second_roi[0], second_roi[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            print(f"Processed {len(processed_info)} vehicles in both ROIs.")
            print(f"Number of vehicles detected: {num_vehicles_detected}")

            # Convert annotated frame to base64
            print("Frame processed successfully.")
            return {
                'processed_info': processed_info,
                'frame_base64': self._convert_frame_to_base64(annotated_frame),
                'num_vehicles_detected': num_vehicles_detected
            }
        else:
            print("No relevant objects detected in the frame.")
            return {
                'processed_info': [{'message': 'No relevant objects detected in the frame'}],
                'frame_base64': self._convert_frame_to_base64(enhanced_frame),
                'num_vehicles_detected': 0
            }



