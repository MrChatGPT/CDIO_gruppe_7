
import time
import threading
import sys
import paho.mqtt.client as mqtt
import uuid
#from pyPS4Controller.controller import Controller 
import math
import json

class MyController():
    '''
    This class is used to define the controller object. It inherits from the Controller class in the pyPS4Controller library.
    The default location of the controller interface is /dev/input/js0.
    '''
    def __init__(self, interface="/dev/input/js0",event_format="3Bh2b", connecting_using_ds4drv=False, **kwargs):
        #super().__init__(interface=interface,
        #                 connecting_using_ds4drv=connecting_using_ds4drv,event_format=event_format, **kwargs)
        self.R3_value = [0, 0]
        self.L3_value = 0
        self.R1_value = 0
        self.L1_value = 0
        self.new_data_callback = None
        self.xy_power = 0
        self.motors = [0,0,0,0]
        self.stick_dead_zone = 0.05
    

    # Function is called when R3 is moved
    def on_R3_x_at_rest(self):
        self.R3_value[0] = 0
        # set the first entries in self.motors[] to the calculated wheel speeds
        self.motors = self.calculate_wheel_speeds(self.R3_value[0], self.R3_value[1], self.L3_value)
        if self.new_data_callback is not None:
            self.new_data_callback() 

    def on_R3_y_at_rest(self):
        self.R3_value[1] = 0 
        self.motors = self.calculate_wheel_speeds(self.R3_value[0], self.R3_value[1], self.L3_value) 
        if self.new_data_callback is not None:
            self.new_data_callback()

    def on_R3_down(self, value):
        self.R3_value[1] = -self.map_stick_value(value)
        self.motors = self.calculate_wheel_speeds(self.R3_value[0], self.R3_value[1], self.L3_value)
        if self.new_data_callback is not None:
            self.new_data_callback()

    def on_R3_up(self, value):
        self.R3_value[1] = -self.map_stick_value(value)
        self.motors = self.calculate_wheel_speeds(self.R3_value[0], self.R3_value[1], self.L3_value)
        if self.new_data_callback is not None:
            self.new_data_callback()

    def on_R3_left(self, value):
        self.R3_value[0] = self.map_stick_value(value)
        self.motors = self.calculate_wheel_speeds(self.R3_value[0], self.R3_value[1], self.L3_value)
        if self.new_data_callback is not None:
            self.new_data_callback()


    def on_R3_right(self, value):
        self.R3_value[0] = self.map_stick_value(value)
        self.motors = self.calculate_wheel_speeds(self.R3_value[0], self.R3_value[1], self.L3_value)
        if self.new_data_callback is not None:
            self.new_data_callback()

    # Function is called when L3 is moved
    def on_L3_x_at_rest(self):
        self.L3_value = 0
        self.motors = self.calculate_wheel_speeds(self.R3_value[0], self.R3_value[1], self.L3_value)
        if self.new_data_callback is not None:
            self.new_data_callback()

    def on_L3_left(self, value):
        self.L3_value = self.map_stick_value(value)
        self.motors = self.calculate_wheel_speeds(self.R3_value[0], self.R3_value[1], self.L3_value)
        if self.new_data_callback is not None:
            self.new_data_callback()

    def on_L3_right(self, value):
        self.L3_value = self.map_stick_value(value)
        self.motors = self.calculate_wheel_speeds(self.R3_value[0], self.R3_value[1], self.L3_value)
        if self.new_data_callback is not None:
            self.new_data_callback()

    # Function is called when R1 is pressed
    def on_R1_press(self):
        self.R1_value = 1
        if self.new_data_callback is not None:
            self.new_data_callback()
    
    def on_R1_release(self):
        self.R1_value = 0
        if self.new_data_callback is not None:
            self.new_data_callback()

    # Function is called when L1 is pressed
    def on_L1_press(self):
        self.L1_value = 1
        if self.new_data_callback is not None:
            self.new_data_callback()
    def on_L1_release(self):
        self.L1_value = 0
        if self.new_data_callback is not None:
            self.new_data_callback()


    def start(self):
        self.listen_thread = threading.Thread(
            target=self.listen, args=(60,), daemon=True)
        self.listen_thread.start()

    def map_stick_value(self,raw_value):
        mapped_value = raw_value / 32767.0
        # Apply dead zone
        if -self.stick_dead_zone < abs(mapped_value) < self.stick_dead_zone:
            return 0
        return mapped_value
   
       

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
    
    @staticmethod
    def calculate_wheel_speeds(x, y, rotation):
        if x == 0 and y == 0:
            if rotation == 0:
                return [0, 0, 0, 0]
            else:
                return [rotation, -rotation, rotation, -rotation]

        angle = math.atan2(y, x)
        xy_speed = math.hypot(x,y)

        wheel_speeds = MyController.calculate_wheel_contributions(angle, xy_speed, rotation)
        wheel_speeds = MyController.normalize_wheel_speeds(wheel_speeds)

        return wheel_speeds

class MQTTClient:
    def __init__(self, broker_url='192.168.1.101', broker_port=1883, topics=None, client_id=None, loop_method="start"):
        self.broker_url = broker_url
        self.broker_port = broker_port
        self.subscribe_topics = topics if topics is not None else []  # Ensure topics is always a list
        self.loop_method = loop_method
        self.client_id = client_id or str(uuid.uuid4())
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, self.client_id)
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
        print(f"Message published: {topic} {message}", flush=True)

    def disconnect(self):
        self.client.loop_stop()  # Stop the loop only if loop_start was used
        self.client.disconnect()
