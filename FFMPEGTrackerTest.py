import os
import subprocess
from VehicleDetectionTracker.VehicleTracker import VehicleTracker

def capture_and_process_frame(camera_url, tracker):
    try:
        capture_command = [
            "ffmpeg",
            "-i", camera_url,
            "-vf", "fps=1",
            "-frames:v", "1",
            "-q:v", "2",
            "pipe:1"  # Use a pipe to send the output to stdout
        ]

        # Execute the FFMPEG command and capture the output in 'output'
        process = subprocess.Popen(capture_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, _ = process.communicate()

        if process.returncode == 0:
            results = tracker.process_frame(output)
        else:
            print(f"Error capturing the image. ffmpeg output: {output}")
    except Exception as e:
        print(f"Error: {e}")

# Usage of the function
# Replace the following variable with your camera URL:
camera_url = "https://49-d2.divas.cloud/CHAN-8294/CHAN-8294_1.stream/playlist.m3u8?83.52.13.30&vdswztokenhash=wTGOQvKvSjyYm7XQplKNtkMpVyLZiBbWZ-bPIwUnXwE="

# Configure the tracker
tracker = VehicleTracker()

# Capture and process a frame
capture_and_process_frame(camera_url, tracker)