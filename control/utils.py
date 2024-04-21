
import time
import threading
import sys
import paho.mqtt.client as mqtt
import uuid 
from pyPS4Controller.controller import Controller 
import math
import json



class MyController(Controller):
    '''
    This class is used to define the controller object. It inherits from the Controller class in the pyPS4Controller library.
    The default location of the controller interface is /dev/input/js0.
    '''
    def __init__(self, interface="/dev/input/js0", connecting_using_ds4drv=False, **kwargs):
        super().__init__(interface=interface,
                         connecting_using_ds4drv=connecting_using_ds4drv, **kwargs)
        self.R3_value = [0, 0]
        self.L3_value = 0
        self.R2_value = 0
        self.x_value = 0
        self.circle_value = 0
        self.new_data_callback = None
        self.angle = 0
        self.power = 0
        self.wheels = [0,0,0,0]
        self.stick_dead_zone = 0.07

    # Function is called when R3 is moved
    def on_R3_x_at_rest(self):
        self.R3_value[0] = 0
        self.wheels, self.angle = self.calc_wheels_speed(self.R3_value[0],self.R3_value[1],self.L3_value,power=self.R2_value)
        if self.new_data_callback is not None:
            self.new_data_callback() 

    def on_R3_y_at_rest(self):
        self.R3_value[1] = 0  
        self.wheels, self.angle = self.calc_wheels_speed(self.R3_value[0],self.R3_value[1],self.L3_value,power=self.R2_value)
        if self.new_data_callback is not None:
            self.new_data_callback()

    def on_R3_down(self, value):
        self.R3_value[1] = -self.map_stick_value(value)
        self.wheels, self.angle = self.calc_wheels_speed(self.R3_value[0],self.R3_value[1],self.L3_value,power=self.R2_value)
        if self.new_data_callback is not None:
            self.new_data_callback()

    def on_R3_up(self, value):
        self.R3_value[1] = -self.map_stick_value(value)
        self.wheels, self.angle = self.calc_wheels_speed(self.R3_value[0],self.R3_value[1],self.L3_value,power=self.R2_value)
        if self.new_data_callback is not None:
            self.new_data_callback()

    def on_R3_left(self, value):
        self.R3_value[0] = self.map_stick_value(value)
        self.wheels, self.angle = self.calc_wheels_speed(self.R3_value[0],self.R3_value[1],self.L3_value,power=self.R2_value)
        if self.new_data_callback is not None:
            self.new_data_callback()


    def on_R3_right(self, value):
        self.R3_value[0] = self.map_stick_value(value)
        self.wheels, self.angle = self.calc_wheels_speed(self.R3_value[0],self.R3_value[1],self.L3_value,power=self.R2_value)
        if self.new_data_callback is not None:
            self.new_data_callback()

    # Function is called when L3 is moved
    def on_L3_x_at_rest(self):
        self.L3_value = 0
        self.wheels, self.angle = self.calc_wheels_speed(self.R3_value[0],self.R3_value[1],self.L3_value,power=self.R2_value)
        if self.new_data_callback is not None:
            self.new_data_callback()

    def on_L3_left(self, value):
        self.L3_value = self.map_stick_value(value)
        self.wheels, self.angle = self.calc_wheels_speed(self.R3_value[0],self.R3_value[1],self.L3_value,power=self.R2_value)
        if self.new_data_callback is not None:
            self.new_data_callback()

    def on_L3_right(self, value):
        self.L3_value = self.map_stick_value(value)
        self.wheels, self.angle = self.calc_wheels_speed(self.R3_value[0],self.R3_value[1],self.L3_value,power=self.R2_value)
        if self.new_data_callback is not None:
            self.new_data_callback()

    # Function is called when R2 is pressed
    def on_R2_press(self, value):
        self.R2_value = (self.map_stick_value(value)+1)/2
        self.wheels, self.angle = self.calc_wheels_speed(self.R3_value[0],self.R3_value[1],self.L3_value,power=self.R2_value)
        if self.new_data_callback is not None:
            self.new_data_callback()
    
    def on_R2_release(self):
        self.R2_value = 0
        self.wheels, self.angle = self.calc_wheels_speed(self.R3_value[0],self.R3_value[1],self.L3_value,power=self.R2_value)
        if self.new_data_callback is not None:
            self.new_data_callback()

    # Function is called when X is pressed
    def on_x_press(self):
        self.x_value = 1
        if self.new_data_callback is not None:
            self.new_data_callback()

    def on_x_release(self):
        self.x_value = 0
        if self.new_data_callback is not None:
            self.new_data_callback()

    def on_circle_press(self):
        self.circle_value = 1
        if self.new_data_callback is not None:
            self.new_data_callback()
    
    def on_circle_release(self):
        self.circle_value = 0
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
    # function to convert x and y values to angle
    def calc_wheels_speed(x, y, rotation, power,angle=None): 
        if x == 0 and y == 0 and rotation == 0:
            return (0, 0, 0, 0), 0
        
        angle = math.atan2(y, x) if angle is None else angle
        sin_angle = math.sin(angle - math.pi/4)
        cos_angle = math.cos(angle - math.pi/4)
        
        max_val = max(abs(sin_angle), abs(cos_angle))
        leftFront = power*(cos_angle/max_val) + rotation #left front
        rightFront = power*(sin_angle/max_val) - rotation #right front
        leftRear = power*(sin_angle/max_val) + rotation #left rear
        rightRear = power*(cos_angle/max_val) - rotation #right rear

        max_speed = max(abs(leftFront), abs(rightFront), abs(leftRear), abs(rightRear))
        if max_speed > 1:
            leftFront = leftFront/(power + abs(rotation))
            rightFront = rightFront/(power + abs(rotation))
            leftRear = leftRear/(power + abs(rotation))
            rightRear = rightRear/(power + abs(rotation))
        return (leftFront, rightFront, leftRear, rightRear), angle

class MQTTClient:
    def __init__(self, broker_url='localhost', broker_port=1883, topics=None, client_id=None, loop_method="start"):
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

if __name__ == "__main__":
    controller = MyController()  # Create the controller object
    controller.start()  # Start the controller thread

    start_time = time.time()
    while time.time() - start_time < 120:  # Run for 10 seconds
        print(f"R3: {controller.R3_value}\nL3: {controller.L3_value}\nAngle: {round(controller.angle,3)}\nPower: {round(controller.power,3)}\nWheels: {[round(w,3) for w in controller.wheels]}\nR2: {controller.R2_value}\nx: {controller.x_value}", flush=True)
        time.sleep(0.1)  # Delay to prevent flooding the output

    sys.exit(0)
