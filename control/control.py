from utils import MQTTClient, MyController

controller = MyController()  # Create the controller object
client = MQTTClient(client_id='controller',loop_method='forever')
topic = 'robot/control'

new_message = None
last_message = None

def publish_controller_data():
    global new_message, last_message  # Declare these variables as global
    new_message = (controller.wheels, controller.x_value, controller.circle_value)
    # Publish wheels data
    if new_message != last_message:
        client.publish(topic, new_message)
        last_message = new_message  # Update the last_message to the new_message after publishing

controller.new_data_callback = publish_controller_data


if __name__ == "__main__":
    controller.start()  
    client.connect()
    while True:
        pass
