from confluent_kafka import Consumer, KafkaError
from VehicleDetectionTracker.VehicleDetectionTracker import VehicleDetectionTracker
import json

conf = {
    'bootstrap.servers': '192.168.1.39:9092"',  # Configure this to your Kafka broker's address.
    'group.id': 'my-group-1',
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
        frame_timestamp = payload.get('frame_timestamp', '')
        frame_data = payload.get('frame_data', '')

        # Process the frame with the tracker
        results = vehicleDetection.process_frame_base64(frame_data, frame_timestamp)
    
        # Optionally, you can access the MAC address and timestamp for further processing
        print(f"MAC Address: {mac_address}")
        print(f"Timestamp: {frame_timestamp}")
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