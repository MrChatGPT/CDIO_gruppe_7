import multiprocessing
from robot_control_utils import MotorController, MQTTClientHandler

def main():
    command_queue = multiprocessing.Queue()

    motor_process = multiprocessing.Process(target=run_motor_controller, args=(command_queue,))
    mqtt_process = multiprocessing.Process(target=run_mqtt_client, args=(command_queue,))

    motor_process.start()
    mqtt_process.start()

    motor_process.join()
    mqtt_process.join()

def run_motor_controller(command_queue):
    motor_controller = MotorController(command_queue)
    motor_controller.run()

def run_mqtt_client(command_queue):
    mqtt_handler = MQTTClientHandler(command_queue)
    mqtt_handler.run()

if __name__ == "__main__":
    main()
