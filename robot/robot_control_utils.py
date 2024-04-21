import paho.mqtt.client as mqtt
import json
import multiprocessing
from gpiozero import Motor, PWMOutputDevice, Device
from gpiozero.pins.pigpio import PiGPIOFactory
import time

class MQTTClientHandler:
    def __init__(self, command_queue, broker_address="192.168.1.101", port=1883):
        self.command_queue = command_queue
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1,"robot_client")
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(broker_address, port, keepalive=60)

    def on_connect(self, client, userdata, flags, rc):
        print(f"Connected with result code {rc}")
        self.client.subscribe("robot/control")

    def on_message(self, client, userdata, message):
        """ Deserialize message and put new speeds in the queue """
        data = json.loads(message.payload.decode("utf-8"))
        self.command_queue.put(data[0])  # Assuming data[0] contains the speeds for the motors

    def run(self):
        """ Start the MQTT client loop """
        self.client.loop_forever()
        

class MotorController:
    def __init__(self, command_queue):
        Device.pin_factory = PiGPIOFactory()

        # Initialize motors
        self.rl = Motor(forward=4, backward=17)
        self.rl_speed = PWMOutputDevice(27)
        self.rr = Motor(forward=22, backward=5)
        self.rr_speed = PWMOutputDevice(6)
        self.fl = Motor(forward=23, backward=24)
        self.fl_speed = PWMOutputDevice(25)
        self.fr = Motor(forward=16, backward=20)
        self.fr_speed = PWMOutputDevice(21)

        self.motors = [(self.fl, self.fl_speed), (self.fr, self.fr_speed),
                       (self.rl, self.rl_speed), (self.rr, self.rr_speed)]
        
        self.command_queue = command_queue

    def run(self):
        """ Process that continuously checks for new motor commands """
        while True:
            speeds = self.command_queue.get()  # Block until a command is received
            print(f"Motor speeds: {speeds} ")
            self.update_motors(speeds)

    def update_motors(self, speeds):
        """ Update the motors' speeds based on received commands """
        for (motor, pwm), speed in zip(self.motors, speeds):
            if speed == 0:
                motor.stop()
            else:
                motor.forward() if speed > 0 else motor.backward()
                pwm.value = abs(speed)
