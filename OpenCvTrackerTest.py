from VehicleDetection.VehicleDetection import VehicleDetection

video_path = "https://35-d2.divas.cloud/CHAN-9477/CHAN-9477_1.stream/playlist.m3u8?83.52.13.30&vdswztokenhash=cTjnsyM47fJ9KyC59bhtuWTbZFVLpfY19HVPg9kIdJM="
vehicle_detection = VehicleDetection()
result_callback = lambda result: print({
    "number_of_vehicles_detected": result["number_of_vehicles_detected"],
    "detected_vehicles": [
        {
            "vehicle_id": vehicle["vehicle_id"],
            "vehicle_type": vehicle["vehicle_type"],
            "detection_confidence": vehicle["detection_confidence"]
        }
        for vehicle in result['detected_vehicles']
    ]
})
vehicle_detection.process_video(video_path, result_callback)
