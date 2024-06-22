from multiprocessing import Process, Queue
from queue import Empty
from turtle import st
from simple_pid import PID
import time
from cam_module.cam2 import camera_process  # Adjust the import as needed
from controller_module.controller import Controller
import math


class ControlLogic:
    def __init__(self, queue, controller, translation_pid, rotation_pid):
        self.queue = queue
        self.controller = controller
        self.pid_translation = translation_pid
        self.pid_rotation = rotation_pid
        self.on_waypoint = False

    def run(self):
        while True:
            # time this loop

            try:
                data = self.queue.get_nowait()
                if data:
                    self.process_data(data)
                else:
                    self.stop_robot()
            except Empty:
                pass
            except Exception as e:
                print(f"Error occurred: {e}")
                break

    def process_data(self, data):
        orange_waypoint = data.get('vector_to_orange_waypoint_robot_frame')
        # print(f"vector to range waypoint: {orange_waypoint}")
        # print(orange_waypoint)
        distance_to_orange_waypoint = data.get(
            'distance_to_closest_orange_waypoint')
        # print(distance_to_orange_waypoint)
        angle_err = data.get('angle_to_closest_orange_ball')
        print(f"Angel error is {angle_err}")

        if orange_waypoint is not None:
            # do pid stuff

            # Calculate the direction angle using the normalized orange_waypoint vector
            direction_angle = math.degrees(math.atan2(
                orange_waypoint[1], orange_waypoint[0]))

            # Determine if the robot is moving straight or diagonally
            if 45 <= abs(direction_angle) <= 135:
                speed_scale = 1  # Scale factor for diagonal movements
            else:
                speed_scale = 0.5  # Scale factor for straight movements

            # Calculate the speed using the distance to the orange_waypoint
            speed = self.pid_translation(
                -distance_to_orange_waypoint) * speed_scale
            y = orange_waypoint[0] * speed
            x = orange_waypoint[1] * speed

            # Calculate the rotation using the angle error
            rotation = self.pid_rotation(angle_err)

            if (distance_to_orange_waypoint > 1):
                rotation = 0
                print(f"x: {x}, y: {y}, rotation: {-rotation}")
                self.controller.publish_control_data(x, y, -rotation)

            elif abs(angle_err) > 1:
                x = y = 0
                self.controller.publish_control_data(x, y, -rotation)

            else:
                self.stop_robot()
                print("Reached orange_waypoint")
                self.on_waypoint = True
        else:
            print("No orange_waypoint")
            # self.stop_robot()

    def stop_robot(self):
        self.controller.publish_control_data(0, 0, 0)


if __name__ == "__main__":
    queue = Queue(maxsize=10)
    video_path = "/dev/video9"

    broker_url = '192.168.1.101'
    broker_port = 1883
    topic = "robot/control"
    controller = Controller(broker_url, broker_port, topic)

    translation_pid = PID(Kp=0.03, Ki=0.00, Kd=0, setpoint=0)
    rotation_pid = PID(Kp=0.01, Ki=0.025, Kd=0.00, setpoint=0)

    translation_pid.output_limits = (0.25, 1)
    rotation_pid.output_limits = (-0.3, 0.3)

    control_flags = {
        'update_orange_balls': False,
        'update_white_balls': False,
        'update_robot': True,
        'update_arena': False
    }

    control_logic = ControlLogic(
        queue, controller, translation_pid, rotation_pid)

    camera_proc = Process(target=camera_process, args=(
        queue, video_path, control_flags))
    camera_proc.start()

    try:
        control_logic.run()
    except KeyboardInterrupt:
        camera_proc.terminate()
        camera_proc.join()
    finally:
        camera_proc.terminate()
        camera_proc.join()
