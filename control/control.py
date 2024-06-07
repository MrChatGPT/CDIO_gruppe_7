
from utils import MQTTClient, MyController
from typing import Tuple, Optional

controller = MyController() 
client = MQTTClient(client_id='controller',loop_method='start')
topic = 'robot/control'

def publish_controller_data(command: Optional[Tuple[float, float, float, int, int]] = None):
    last_message = None  

    def publish():
        nonlocal last_message
        new_message = (controller.motors, controller.R1_value, controller.L1_value)
        #print(controller.R3_value, controller.L3_value)
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


# 