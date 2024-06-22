import time
import paho.mqtt.client as mqtt
import math
import json
import uuid


class MQTTClient:
    def __init__(self, broker_url='192.168.1.101', broker_port=1883, topics=None, client_id=None, loop_method="start"):
        self.broker_url = broker_url
        self.broker_port = broker_port
        # Ensure topics is always a list
        self.subscribe_topics = topics if topics is not None else []
        self.loop_method = loop_method
        self.client_id = client_id or str(uuid.uuid4())
        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION1, self.client_id)
        self.client.reconnect_delay_set(min_delay=1, max_delay=120)

        # Assign event callbacks
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def connect(self):
        self.client.connect(self.broker_url, self.broker_port, 60)
        if self.loop_method == "start":
            self.client.loop_start()  # Start the network loop in a separate thread
        elif self.loop_method == "forever":
            self.client.loop_forever()  # Blocks here

    def on_connect(self, client, userdata, flags, rc):
        print(f"Connected with result code {rc}")
        # Subscribing to all topics
        for topic in self.subscribe_topics:
            self.client.subscribe(topic)

    def on_message(self, client, userdata, msg):
        print(f"Message received: {msg.topic} {msg.payload.decode()}")

    def publish(self, topic, message):
        # use json.dumps to serialize dictionary to a JSON formatted string
        self.client.publish(topic, json.dumps(message))
        # print(f"Message published: {topic} {message}", flush=True)

    def disconnect(self):
        self.client.loop_stop()  # Stop the loop only if loop_start was used
        self.client.disconnect()


class Controller:

    def __init__(self, broker_url, broker_port, topic):
        self.mqtt_client = MQTTClient(broker_url, broker_port)
        self.mqtt_client.connect()
        self.topic = topic

    @staticmethod
    def calculate_wheel_contributions(angle, xy_power, rotation):
        adjusted_angle = angle - math.pi / 4
        sin_angle = math.sin(adjusted_angle)
        cos_angle = math.cos(adjusted_angle)
        max_val = max(abs(sin_angle), abs(cos_angle))

        left_front = xy_power * (cos_angle / max_val) + rotation
        right_front = xy_power * (sin_angle / max_val) - rotation
        left_rear = xy_power * (sin_angle / max_val) + rotation
        right_rear = xy_power * (cos_angle / max_val) - rotation

        return [left_front, right_front, left_rear, right_rear]

    @staticmethod
    def normalize_wheel_speeds(speeds):
        max_speed = max(abs(speed) for speed in speeds)
        if max_speed > 1:
            return [speed / max_speed for speed in speeds]
        return speeds

    def calculate_wheel_speeds(self, x, y, rotation):

        if x == 0 and y == 0:
            if rotation == 0:
                return [0, 0, 0, 0]
            else:
                return [rotation, -rotation, rotation, -rotation]

        angle = math.atan2(y, x)
        xy_speed = math.hypot(x, y)

        wheel_speeds = Controller.calculate_wheel_contributions(
            angle, xy_speed, rotation)
        wheel_speeds = Controller.normalize_wheel_speeds(wheel_speeds)

        return wheel_speeds

    def publish_control_data(self, x, y, rotation, int1=0, int2=0):
        wheel_speeds = self.calculate_wheel_speeds(x, y, rotation)
        control_data = (wheel_speeds, int1, int2)
        self.mqtt_client.publish(self.topic, control_data)


# Example usage
if __name__ == "__main__":
    controller = Controller(broker_url='192.168.1.101',
                            broker_port=1883, topic="robot/control")

    # Simulate joystick input
    # x, y, rotation = 0.5, 0.5, 0.1
    com_right = (0.5, 0, 0, 0, 0)
    com_left = (-0.5, 0, 0, 0, 0)
    com_forward = (0, 0.5, 0, 0, 0)
    com_backward = (0, -0.5, 0, 0, 0)
    com_rotate_right = (0, 0, 0.5, 0, 0)
    com_rotate_left = (0, 0, -0.5, 0, 0)
    com_stop = (0, 0, 0, 0, 0)

    while True:
        controller.publish_control_data(*com_forward)
        time.sleep(1)  # Publish control data every second
        controller.publish_control_data(*com_stop)
        time.sleep(1)
