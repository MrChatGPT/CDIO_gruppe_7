from utils import MQTTClient, MyController
import time
import sys
controller = MyController()  # Create the controller object
client = MQTTClient(client_id='controller')
topics ={'wheels':'robot/wheels', 'servo':'robot/servo_motor', 'stepper':'robot/stepper_motor'}

def publish_controller_data():
    # Publish wheels data
    client.publish(topics['wheels'], controller.wheels)
    # Publish servo data (assuming this is another attribute you want to monitor)
    client.publish(topics['servo'], controller.servo)
    # Publish stepper motor data
    client.publish(topics['stepper'], controller.stepper)

controller.new_data_callback = publish_controller_data


if __name__ == "__main__":
    controller.start()  # Start the controller thread
    client.connect()
    #client.publish(topics['wheels'], controller.wheels)
    while True:
        time.sleep(1)
