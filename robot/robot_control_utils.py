import paho.mqtt.client as mqtt
import json
from gpiozero import Motor, PWMOutputDevice, Device, DigitalInputDevice
from gpiozero.pins.pigpio import PiGPIOFactory
import time
import threading

class Encoder(object):
    def __init__(self, pin, alpha=0.1, stop_threshold=0.02):
        # Encoder setup
        self._value = 0
        self.encoder = DigitalInputDevice(pin)
        self.encoder.when_activated = self._increment
        self.encoder.when_deactivated = self._increment
        
        # Time tracking
        self.last_time = time.time()
        
        # Speed calculation variables
        self.alpha = alpha  # Smoothing factor for EMA
        self.average_speed = 0  # Initialize the EMA speed
        self.stop_threshold = stop_threshold  # Time threshold in seconds to consider the motor stopped

    def _increment(self):
        """ Increment the encoder count and update the speed using EMA. """
        self._value += 1
        current_time = time.time()
        time_interval = current_time - self.last_time
        
        if time_interval > 0:
            # Calculate speed in revolutions per minute (RPM)
            instant_speed = 1 / (time_interval * 40) * 60  # Assuming 40 counts per revolution
            # Update the EMA of speed
            self.average_speed = (self.alpha * instant_speed) + ((1 - self.alpha) * self.average_speed)
        
        self.last_time = current_time

    def reset(self):
        """ Reset the encoder count and last time tracking. """
        self._value = 0
        self.last_time = time.time()
        self.average_speed = 0

    def value(self):
        """ Return the current encoder count. """
        return self._value

    def get_speed(self):
        """ Return the current average speed (RPM). If no increment recently, return 0. """
        if time.time() - self.last_time > self.stop_threshold:
            return 0  # Return 0 RPM if there have been no increments in the past threshold seconds
        return self.average_speed


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
        self.command_queue.put(data)

    def run(self):
        """ Start the MQTT client loop """
        self.client.loop_forever()
class MotorController:
    def __init__(self, command_queue, encoder_pin=18):
        Device.pin_factory = PiGPIOFactory()
        # Initialize motors and PWM controllers
        self.rl = Motor(forward=4, backward=17)
        self.rl_speed = PWMOutputDevice(27)
        self.rr = Motor(forward=22, backward=5)
        self.rr_speed = PWMOutputDevice(6)
        self.fl = Motor(forward=23, backward=24)
        self.fl_speed = PWMOutputDevice(25)
        self.fr = Motor(forward=16, backward=20)
        self.fr_speed = PWMOutputDevice(21)
        self.pickup = Motor(forward=7, backward=8)
        self.pickup_speed = PWMOutputDevice(13)
        self.motors = [(self.fl, self.fl_speed), (self.fr, self.fr_speed),
                       (self.rl, self.rl_speed), (self.rr, self.rr_speed),
                       (self.pickup, self.pickup_speed)]
        self.command_queue = command_queue
        self.encoder = Encoder(encoder_pin)
        self.encoder_thread = None

    def run_motor_with_encoder(self, target_ticks, initial_direction, speed, sleep_duration):
        """ Run the pickup motor in a specified direction and speed until the encoder reaches the target tick count or an obstruction is detected. """
        self.encoder.reset()
        motor_action = self.pickup.forward if initial_direction == "forward" else self.pickup.backward
        motor_action()
        self.pickup_speed.value = speed 

        try:
            while self.encoder.value() < target_ticks:
                time.sleep(sleep_duration)
                if self.encoder.get_speed() == 0 and self.encoder.value() > 1: 
                    self.backtrack_motor(self.encoder.value() - 7, speed, sleep_duration) 
                    break
        finally:
            self.pickup.stop()

    def backtrack_motor(self, steps, speed, sleep_duration):
        """ Reverse the motor to return to the starting position by the same number of steps it moved forward. """
        self.pickup.reverse()
        self.pickup_speed.value = speed  # Use the same speed for backtracking
        self.encoder.reset()

        while self.encoder.value() < steps:
            time.sleep(sleep_duration)  # Short delay to allow for encoder count updates

        self.pickup.stop()

    def run(self):
        """ Continuously checks for new motor commands. """
        while True:
            command = self.command_queue.get()  # Block until a command is received
            self.update_motors(command)
            print(f"Command: {command}")

    def update_motors(self, command):
        """ Update the motors' speeds based on received commands. """
        for (motor, pwm), speed in zip(self.motors, command[0]):
            if speed == 0:
                motor.stop()
            else:
                motor.forward() if speed > 0 else motor.backward()
                pwm.value = abs(speed)

        if command[1] == 1:  
            if self.encoder_thread is None or not self.encoder_thread.is_alive():
                self.encoder_thread = threading.Thread(target=self.run_motor_with_encoder, args=(49, "forward", 0.7, 0.0001))
                self.encoder_thread.start()
        else:
                if command[2] == 1:
                        if self.encoder_thread is None or not self.encoder_thread.is_alive():
                                self.encoder_thread = threading.Thread(target=self.run_motor_with_encoder, args=(500, "backward", 1, 0.0001))
                                self.encoder_thread.start()
                        
                
                
                
                
                
                
                

