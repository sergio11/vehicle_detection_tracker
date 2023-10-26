import cv2
import base64
from PIL import Image
from io import BytesIO
from VehicleDetectionTracker.VehicleTracker import VehicleTracker
import os

# Crea el directorio "outputs_images" si no existe
output_dir = "outputs_images"
os.makedirs(output_dir, exist_ok=True)

video_path = "https://49-d2.divas.cloud/CHAN-8294/CHAN-8294_1.stream/playlist.m3u8?83.52.13.30&vdswztokenhash=wTGOQvKvSjyYm7XQplKNtkMpVyLZiBbWZ-bPIwUnXwE="
cap = cv2.VideoCapture(video_path)
tracker = VehicleTracker()
frame_number = 0

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = tracker.process_frame(frame)
        print(f"Number of Vehicles Detected: {results['num_vehicles_detected']}")

        if results['frame_base64']:
            frame_base64 = results['frame_base64']
            frame_bytes = base64.b64decode(frame_base64)
            frame_image = Image.open(BytesIO(frame_bytes))
            frame_image.save(os.path.join(output_dir, f"frame_{frame_number}.jpg"))

        for info in results['processed_info']:
            print(f"Direction: {info['direction']}")
            print(f"Color: {info['color']}")
            print(f"Timestamp: {info['timestamp']}")
            print(f"License Plate: {info['license_plate']}")

            if info['vehicle_image_base64']:
                vehicle_image_base64 = info['vehicle_image_base64']
                vehicle_bytes = base64.b64decode(vehicle_image_base64)
                vehicle_image = Image.open(BytesIO(vehicle_bytes))
                vehicle_image.save(os.path.join(output_dir, f"vehicle_{frame_number}_{info['direction']}.jpg"))

        frame_number += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        print(f"Open Video failed")
        break

cap.release()
cv2.destroyAllWindows()