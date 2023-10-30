from VehicleDetectionTracker.VehicleDetectionTracker import VehicleDetectionTracker

# Approximate values for traffic cameras
focal_length_mm = 8.0  # Focal length in millimeters
sensor_size_mm = 1/3  # Sensor size in inches
video_path = "https://49-d2.divas.cloud/CHAN-8293/CHAN-8293_1.stream/playlist.m3u8?77.211.5.136&vdswztokenhash=mzG-kruh8fv5WKEjXYCT0Tlbc23hSuoDRXidQ-aE0zk="
vehicle_detection = VehicleDetectionTracker()
result_callback = lambda result: print({
    "number_of_vehicles_detected": result["number_of_vehicles_detected"],
    "detected_vehicles": [
        {
            "vehicle_id": vehicle["vehicle_id"],
            "vehicle_type": vehicle["vehicle_type"],
            "detection_confidence": vehicle["detection_confidence"],
            "color_info": vehicle["color_info"],
            "model_info": vehicle["model_info"],
            "speed_kph": vehicle["speed_kph"]
        }
        for vehicle in result['detected_vehicles']
    ]
})
vehicle_detection.process_video(video_path, result_callback = result_callback, focal_length_mm=focal_length_mm, sensor_size_mm=sensor_size_mm)
