from VehicleDetectionTracker.VehicleDetectionTracker import VehicleDetectionTracker

video_path = "[[YOUR_STREAMING_SOURCE]]"
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
            "speed_info": vehicle["speed_info"]
        }
        for vehicle in result['detected_vehicles']
    ]
})
vehicle_detection.process_video(video_path, result_callback = result_callback)
