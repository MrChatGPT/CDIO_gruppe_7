
from time import sleep
from utils import MQTTClient, MyController
from typing import Tuple, Optional

controller = MyController() 
client = MQTTClient(client_id='controller',loop_method='start')
topic = 'robot/control'

def publish_controller_data(command: Optional[Tuple[float, float, float, int, int]] = None):
    '''
    This function will be called by the controller whenever it has new data to publish.
    Alternatively, it can be called with a command to update the controller's state.
    The command is a tuple of the form (x, y, rotation, R1_value, L1_value).
    x, y, and rotation are floats between -1 and 1, and R1_value and L1_value are either 0 or 1.
    R1_value is ball intake, L1_value is ball eject.
    '''
    last_message = None

    def update_controller(command: Tuple[float, float, float, int, int]):
        controller.motors = controller.calculate_wheel_speeds(command[0], command[1], command[2])
        controller.R1_value = command[3]
        controller.L1_value = command[4]
    
    def publish():
        nonlocal last_message
        
        new_message = (controller.motors, controller.R1_value, controller.L1_value)
        #print(controller.R3_value, controller.L3_value)
        
        if new_message != last_message: # Only publish if message changed.
            client.publish(topic, new_message)
            last_message = new_message

    if command is not None:
        update_controller(command)
        publish()
    
    return publish


controller.new_data_callback = publish_controller_data()


if __name__ == "__main__":
    controller.start()  
    client.connect()
    while True:
        