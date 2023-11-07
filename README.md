# 🚗 VehicleDetectionTracker: Real-time vehicle detection and tracking powered by YOLO. 🚙🚕

Effortlessly track and detect vehicles in images and videos using state-of-the-art YOLO object detection and tracking, powered by Ultralytics. Boost your computer vision project with the VehicleDetectionTracker, a versatile Python package that simplifies vehicle tracking and detection in a variety of applications. 🚙🚕

- Detect vehicles in real-time or from pre-recorded videos.
- Accurately track vehicles' positions.
- Brand and color classification. The classifiers are based on MobileNet v3 (Alibaba MNN backend).
- Empower traffic analysis, automated surveillance, and more.
- Harness the capabilities of YOLO for precise object detection.

Whether you're working on traffic management, video analysis, or machine learning projects, the VehicleDetectionTracker provides the tools you need to enhance your results. Explore detailed documentation and examples on the [GitHub repository](https://github.com/sergio11/vehicle_detection_tracker), and get started with vehicle tracking in no time!

[![GitHub](https://img.shields.io/badge/GitHub-View%20on%20GitHub-blue?style=flat-square)](https://github.com/sergio11/vehicle_detection_tracker)
[![PyPI](https://img.shields.io/pypi/v/VehicleDetectionTracker.svg?style=flat-square)](https://pypi.org/project/VehicleDetectionTracker/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://github.com/sergio11/vehicle_detection_tracker/blob/main/LICENSE)

## Features 🌟

- Efficient vehicle detection and tracking in images and videos.
- Vehicle color estimation.
- Easy integration into your computer vision projects.

## Installation 🚀

You can easily install VehicleDetectionTracker using pip:

```bash
pip install VehicleDetectionTracker
```

## Usage 📷

You can quickly get started with VehicleDetectionTracker to detect and track vehicles in images and videos. Below are two usage examples, each tailored to different scenarios:

### Example 1: Real-Time Video Stream (OpenCV)

This example demonstrates how to use VehicleDetectionTracker to process a real-time video stream using OpenCV. Simply provide the URL of the video stream to get started:

```python
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
            "model_info": vehicle["model_info"]
        }
        for vehicle in result['detected_vehicles']
    ]
})
vehicle_detection.process_video(video_path, result_callback)
```

###  Example 2: Kafka Topic Processing

For more advanced use cases, VehicleDetectionTracker can also be integrated with Apache Kafka for processing video frame by frame:

```python
from confluent_kafka import Consumer, KafkaError
from VehicleDetectionTracker.VehicleDetectionTracker import VehicleDetectionTracker
import json

conf = {
    'bootstrap.servers': '192.168.1.39:9092"',  # Configure this to your Kafka broker's address.
    'group.id': 'my-group',
    'auto.offset.reset': 'earliest'
}

topic = 'iot-camera-frames'
consumer = Consumer(conf)
consumer.subscribe([topic])

vehicleDetection = VehicleDetectionTracker()

while True:
    msg = consumer.poll(1.0)

    if msg is None:
        continue
    if msg.error():
        if msg.error().code() == KafkaError._PARTITION_EOF:
            print('End of partition, message read: {} [{}] at offset {}'.format(
                msg.key(), msg.partition(), msg.offset()))
        else:
            print('Kafka error: {}'.format(msg.error()))
    else:
        # Process the message (a frame) received from Kafka
        payload = json.loads(msg.value())
        
        mac_address = payload.get('mac_address', '')
        timestamp = payload.get('timestamp', '')
        frame_data = payload.get('frame_data', '')

        # Process the frame with the tracker
        results = vehicleDetection.process_frame_base64(frame_data)
    
        # Optionally, you can access the MAC address and timestamp for further processing
        print(f"MAC Address: {mac_address}")
        print(f"Timestamp: {timestamp}")
        print({
            "number_of_vehicles_detected": results.get("number_of_vehicles_detected", 0),
            "detected_vehicles": [
                {
                    key: vehicle.get(key, None)
                    for key in ["vehicle_id", "vehicle_type", "detection_confidence"]
                }
                for vehicle in results.get('detected_vehicles', [])
            ]
        })

consumer.close()
```

These examples showcase the flexibility of VehicleDetectionTracker and its ability to adapt to various real-world scenarios. Explore the repository's documentation and examples for more in-depth guidance.

### **Screenshots:** Here are some screenshots that demonstrate the functionality of VehicleDetectionTracker:

![Screenshot 1](screenshots/screenshot_1.PNG) 
![Screenshot 2](screenshots/screenshot_2.PNG)
![Screenshot 3](screenshots/screenshot_3.PNG)
![Screenshot 4](screenshots/screenshot_4.PNG)
![Screenshot 5](screenshots/screenshot_5.PNG)
![Screenshot 6](screenshots/screenshot_6.PNG)

## Documentation 📚

Detailed documentation and examples are available on the [GitHub repository](https://github.com/sergio11/vehicle_detection_tracker).

## License 📜

This project is licensed under the MIT License - see the [LICENSE](https://github.com/sergio11/vehicle_detection_tracker/blob/main/LICENSE) file for details.

## Acknowledgments 🙏

- This package is powered by [YOLO](https://github.com/ultralytics/yolov5) for object detection.
- Special thanks to the open-source community for their contributions.

## Get in Touch 📬

If you have any questions, feedback, or suggestions, feel free to reach out at [dreamsoftware92@gmail.com](mailto:dreamsoftware92@gmail.com).

## Happy Tracking! 🚀👁️

[![GitHub](https://img.shields.io/badge/GitHub-View%20on%20GitHub-blue?style=flat-square)](https://github.com/sergio11/vehicle_detection_tracker)
[![PyPI](https://img.shields.io/pypi/v/VehicleDetectionTracker.svg?style=flat-square)](https://pypi.org/project/VehicleDetectionTracker/)
