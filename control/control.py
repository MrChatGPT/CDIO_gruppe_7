from calendar import c
from utils import MQTTClient, MyController
from time import sleep

controller = MyController() 
client = MQTTClient(client_id='controller',loop_method='start')
topic = 'robot/control'

def publish_controller_data():
    last_message = None  

    def publish():
        nonlocal last_message
        new_message = controller.motors,controller.R1_value,controller.L1_value
        if new_message != last_message: # Only publish if message changed.
            client.publish(topic, new_message)
            last_message = new_message
    
    return publish


controller.new_data_callback = publish_controller_data()


if __name__ == "__main__":
    controller.start()  
    client.connect()
    while True:
        pass


