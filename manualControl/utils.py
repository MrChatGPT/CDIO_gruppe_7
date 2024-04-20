
import time
import threading
import sys
import paho.mqtt.client as mqtt
import uuid 
from pyPS4Controller.controller import Controller



class MyController(Controller):
    '''
    This class is used to define the controller object. It inherits from the Controller class in the pyPS4Controller library.
    The default location of the controller interface is /dev/input/js0.
    '''
    def __init__(self, interface="/dev/input/js0", connecting_using_ds4drv=False, **kwargs):
        super().__init__(interface=interface,
                         connecting_using_ds4drv=connecting_using_ds4drv, **kwargs)
        self.R3_value = [0, 0]
        self.L3_value = [0, 0]
        self.R1_value = 0
        self.x_value = 0

    # Function is called when R3 is moved
    def on_R3_x_at_rest(self):
        self.R3_value[0] = 0

    def on_R3_y_at_rest(self):
        self.R3_value[1] = 0

    def on_R3_down(self, value):
        self.R3_value[1] = self.map_stick_value(value)

    def on_R3_up(self, value):
        self.R3_value[1] = self.map_stick_value(value)

    def on_R3_left(self, value):
        self.R3_value[0] = self.map_stick_value(value)

    def on_R3_right(self, value):
        self.R3_value[0] = self.map_stick_value(value)

    # Function is called when L3 is moved
    def on_L3_x_at_rest(self):
        self.L3_value[0] = 0

    def on_L3_y_at_rest(self):
        self.L3_value[1] = 0

    def on_L3_down(self, value):
        self.L3_value[1] = self.map_stick_value(value)

    def on_L3_up(self, value):
        self.L3_value[1] = self.map_stick_value(value)

    def on_L3_left(self, value):
        self.L3_value[0] = self.map_stick_value(value)

    def on_L3_right(self, value):
        self.L3_value[0] = self.map_stick_value(value)

    # Function is called when R1 is pressed
    def on_R1_press(self):
        self.R1_value = 1

    def on_R1_release(self):
        self.R1_value = 0

    # Function is called when X is pressed
    def on_x_press(self):
        self.x_value = 1

    def on_x_release(self):
        self.x_value = 0

    def start(self):
        self.listen_thread = threading.Thread(
            target=self.listen, args=(60,), daemon=True)
        self.listen_thread.start()

    @staticmethod
    def map_stick_value(raw_value):
        return raw_value / 32767.0



class MQTTClient:
    def __init__(self, broker_url='localhost', broker_port=1883, topics=None, client_id=None, loop_method="start"):
        self.broker_url = broker_url
        self.broker_port = broker_port
        self.subscribe_topics = topics if topics is not None else []  # Ensure topics is always a list
        self.loop_method = loop_method
        self.client_id = client_id or str(uuid.uuid4())
        self.client = mqtt.Client(client_id=self.client_id)
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
        self.client.publish(topic, message)

    def disconnect(self):
        self.client.loop_stop()  # Stop the loop only if loop_start was used
        self.client.disconnect()

# Usage
if __name__ == "__main__":
    topics = ["test/topic"]  # Example with a single topic in a list
    client = MQTTClient(broker_url="localhost", broker_port=1883, topics=topics)
    client.connect()
    time.sleep(1)  # Small delay to ensure connection setup
    client.publish(topics[0], "Hello MQTT!")
    time.sleep(10)  # Keep the client running to listen for messages
    client.disconnect()


if __name__ == "__main__":
    controller = MyController()  # Create the controller object
    controller.start()  # Start the controller thread

    start_time = time.time()
    while time.time() - start_time < 60:  # Run for 10 seconds
        print(f"R3: {controller.R3_value}, L3: {controller.L3_value}, R1: {controller.R1_value}, x: {controller.x_value}", flush=True)
        time.sleep(0.1)  # Delay to prevent flooding the output

    sys.exit(0)
