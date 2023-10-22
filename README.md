# VehicleDetectionTracker 🚗

Effortlessly track and detect vehicles in images and videos using state-of-the-art YOLO object detection and tracking, powered by Ultralytics. Boost your computer vision project with the VehicleDetectionTracker, a versatile Python package that simplifies vehicle tracking and detection in a variety of applications. 🚙🚕

- Detect vehicles in real-time or from pre-recorded videos.
- Accurately track vehicles' positions, speeds, and directions.
- Enjoy flexibility with customizable speed thresholds and color recognition.
- Empower traffic analysis, automated surveillance, and more.
- Harness the capabilities of YOLO for precise object detection.

Whether you're working on traffic management, video analysis, or machine learning projects, the VehicleDetectionTracker provides the tools you need to enhance your results. Explore detailed documentation and examples on the [GitHub repository](https://github.com/sergio11/vehicle_detection_tracker), and get started with vehicle tracking in no time!

[![GitHub](https://img.shields.io/badge/GitHub-View%20on%20GitHub-blue?style=flat-square)](https://github.com/sergio11/vehicle_detection_tracker)
[![PyPI](https://img.shields.io/pypi/v/VehicleDetectionTracker.svg?style=flat-square)](https://pypi.org/project/VehicleDetectionTracker/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://github.com/sergio11/vehicle_detection_tracker/blob/main/LICENSE)

## Features 🌟

- Efficient vehicle detection and tracking in images and videos.
- Speed measurement for tracked vehicles.
- Vehicle color estimation.
- Direction of vehicle movement detection.
- Easy integration into your computer vision projects.

## Installation 🚀

You can easily install VehicleDetectionTracker using pip:

```bash
pip install VehicleDetectionTracker
```

## Usage 📷

```python
from VehicleDetectionTracker import VehicleDetectionTracker

# Create a VehicleDetectionTracker instance
tracker = VehicleDetectionTracker()

# Process a frame
frame = cv2.imread('your_frame.jpg')
result = tracker.process_frame(frame)

# Access processed information
if result:
    processed_info = result['processed_info']
    annotated_frame_base64 = result['frame_base64']
    # Your custom processing logic here

```

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
