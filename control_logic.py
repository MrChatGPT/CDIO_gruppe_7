from multiprocessing import Process, Queue, Manager
from queue import Empty
import os
import sys
from simple_pid import PID
import time
from cam_module.cam2 import camera_process  # Adjust the import as needed
from controller_module.controller import Controller
import math


class ControlLogic:
    def __init__(self, queue, controller, translation_pid, rotation_pid, control_flags):
        self.queue = queue
        self.controller = controller
        self.pid_translation = translation_pid
        self.pid_rotation = rotation_pid
        self.on_waypoint = False
        self.on_ball = False
        self.ball_collected = False
        self.distance_tolerance = 2
        self.angle_tolerance = 2
        self.pid_scaling_factor = 0.5
        self.control_flags = control_flags
        self.ball_count = 0
        self.to_goal = False
        self.arena_check = False
        self.on_goal = False

    def run(self):
        while True:
            try:
                data = self.queue.get_nowait()
                if data:
                    if not self.to_goal:
                        self.collect_ball(data, color='white')
                    else:
                        self.score_ball(data)
                else:
                    self.stop_robot()
            except Empty:
                pass
            except Exception as e:
                print(f"Error occurred: {e}")
                break

    def collect_ball(self, data, color):
        # Get the initial waypoint data
        vector_waypoint = data.get(f'vector_to_{color}_waypoint_robot_frame')
        distance_to_waypoint = data.get(f'distance_to_closest_{color}_waypoint')
        angle_err_to_waypoint = data.get(f'angle_to_closest_{color}_waypoint')
        waypoints = data.get('arena_data')

       
        # get ball data
        vector_ball = data.get(f'vector_to_{color}_ball_robot_frame')
        distance_to_ball = data.get(f'distance_to_closest_{color}_ball')
        angle_err_to_ball = data.get(f'angle_to_closest_{color}_ball')

        # get critical length data
        robot_critical_length = data.get('robot_critical_length')

        # print data
        print(f"Distance to {color}_waypoint: {distance_to_waypoint}")
        print(f"Angle error to {color}_waypoint: {angle_err_to_waypoint}")
        print(f"Distance to {color}_ball: {distance_to_ball}")
        print(f"Angle error to {color}_ball: {angle_err_to_ball}")
        print('control flags:', self.control_flags)

         # If the initial waypoint data is invalid, find the closest valid waypoint
        if (len(vector_waypoint) == 0):
            print("no balls reachable\n waypoint info = ", vector_waypoint)
            self.arena_check = True
            closest_waypoint = None
            min_distance = float('inf')
            
            for waypoint in waypoints:
                vector, distance, angle = waypoint
                
                if distance < min_distance:
                    min_distance = distance
                    closest_waypoint = waypoint
            
            if closest_waypoint:
                vector_waypoint, distance_to_waypoint, angle_err_to_waypoint = closest_waypoint

            # Check if the current waypoint matches any waypoint, and find the next closest one
            current_waypoint = (vector_waypoint, distance_to_waypoint, angle_err_to_waypoint)
            waypoints_sorted_by_distance = sorted(waypoints, key=lambda wp: wp[1])
            
            for wp in waypoints_sorted_by_distance:
                if wp != current_waypoint:
                    vector_waypoint, distance_to_waypoint, angle_err_to_waypoint = wp
                    break

        # print(data)
        if(not self.arena_check):
            try:
                if not self.on_waypoint:
                    # reset control flags to false except for update_robot
                    for key in self.control_flags.keys():
                        if key != 'update_robot':
                            self.control_flags[key] = False

                    if distance_to_waypoint > self.distance_tolerance:

                        # Calculate the direction angle using the normalized orange_waypoint vector
                        direction_angle = math.degrees(math.atan2(
                            vector_waypoint[1], vector_waypoint[0]))

                        # Determine if the robot is moving straight or diagonally
                        if 45 <= abs(direction_angle) <= 135:
                            speed_scale = 1  # Scale factor for diagonal movements
                        else:
                            speed_scale = 0.5  # Scale factor for straight movements

                        # Calculate the speed using the distance to the waypoint
                        speed = self.pid_translation(
                            -distance_to_waypoint) * speed_scale
                        y = vector_waypoint[0] * speed
                        x = vector_waypoint[1] * speed

                        # set rotation to 0
                        rotation = 0
                        self.controller.publish_control_data(x, y, rotation)

                    else:
                        self.stop_robot()
                        print(f"Reached {color}_waypoint")

                        time.sleep(1)
                        self.on_waypoint = True

                elif abs(angle_err_to_ball) > self.angle_tolerance:
                    rotation = -self.pid_rotation(angle_err_to_ball)

                    # set x and y to 0
                    x = y = 0

                    self.controller.publish_control_data(x, y, rotation)

                elif distance_to_ball - robot_critical_length > self.distance_tolerance:
                    # scale pid constants down for the last part of the movement by 0.5
                    self.pid_translation.Kp = self.pid_translation.Kp * self.pid_scaling_factor
                    self.pid_translation.Ki = self.pid_translation.Ki * self.pid_scaling_factor
                    self.pid_translation.Kd = self.pid_translation.Kd * self.pid_scaling_factor

                    # Calculate the direction angle using the normalized orange_ball vector
                    direction_angle = math.degrees(math.atan2(
                        vector_ball[1], vector_ball[0]))

                    # Determine if the robot is moving straight or diagonally
                    if 45 <= abs(direction_angle) <= 135:
                        speed_scale = 1

                    else:
                        speed_scale = 0.5

                    # Calculate the speed using the distance to the ball
                    speed = self.pid_translation(
                        -(distance_to_ball - robot_critical_length)) * speed_scale

                    y = vector_ball[0] * speed

                    x = vector_ball[1] * speed

                    # set rotation to 0

                    rotation = 0
                    self.controller.publish_control_data(x, y, rotation)

                else:
                    # reset pid constants
                    self.pid_translation.Kp = self.pid_translation.Kp / self.pid_scaling_factor
                    self.pid_translation.Ki = self.pid_translation.Ki / self.pid_scaling_factor
                    self.pid_translation.Kd = self.pid_translation.Kd / self.pid_scaling_factor
                    # Collect the ball
                    self.ball_count = self.ball_count+1
                    # get data:
                    white_balls = data.get(f'white_ball_centers')
                    blocked_balls = data.get(f'blocked_white_centers')
                    print(f"picking up {color} ball")
                    self.ball_in()
                    self.stop_robot()

                    if (self.ball_count % 4 == 0) or ((len(white_balls) == 0) and (len(blocked_balls) == 0)):
                        self.to_goal = True
                    # set control flags to true
                    for key in self.control_flags.keys():
                        self.control_flags[key] = True

                    print('Uddating arena, robot and balls')
                    print('control flags:', self.control_flags)
                    time.sleep(5)
                    self.on_waypoint = False

            except Exception as e:
                print(f"Error occurred in collect_ball method: {e}")
                print('Stopping robot')
                self.stop_robot()
        else:
            try:
                for key in self.control_flags.keys():
                        if key != 'update_robot':
                            self.control_flags[key] = False

                if distance_to_waypoint > self.distance_tolerance:

                    # Calculate the direction angle using the normalized orange_waypoint vector
                    direction_angle = math.degrees(math.atan2(
                        vector_waypoint[1], vector_waypoint[0]))

                    # Determine if the robot is moving straight or diagonally
                    if 45 <= abs(direction_angle) <= 135:
                        speed_scale = 1  # Scale factor for diagonal movements
                    else:
                        speed_scale = 0.5  # Scale factor for straight movements

                    # Calculate the speed using the distance to the waypoint
                    speed = self.pid_translation(
                        -distance_to_waypoint) * speed_scale
                    y = vector_waypoint[0] * speed
                    x = vector_waypoint[1] * speed

                    # set rotation to 0
                    rotation = 0
                    self.controller.publish_control_data(x, y, rotation)

                else:
                    self.stop_robot()
                    print(f"Reached {color}_waypoint")
                    time.sleep(1)
                    self.arena_check = False

            except Exception as e:
                print(f"Error occurred in arena_waypoint: {e}")
                print('Stopping robot')
                self.stop_robot()

            

    def stop_robot(self):
        self.controller.publish_control_data(0, 0, 0)

    def ball_in(self):
        self.controller.publish_control_data(0, 0, 0, 1, 0)

    def ball_out(self):
        self.controller.publish_control_data(0, 0, 0, 0, 1)

    def score_ball(self, data):

        waypoints = data.get(f'arena_data')
        vectorone_waypoint = waypoints[2][0]
        vectortwo_waypoint = waypoints[5][0]

        distanceone_to_waypoint = waypoints[2][1]
        distancetwo_to_waypoint = waypoints[5][1]

        angleone_err_to_waypoint = waypoints[2][2]
        angletwo_err_to_waypoint = waypoints[5][2]

        if (distanceone_to_waypoint < distancetwo_to_waypoint):
            vector_waypoint = vectorone_waypoint
            distance_to_waypoint = distanceone_to_waypoint
            angle_err_to_waypoint = angleone_err_to_waypoint
        else:
            vector_waypoint = vectortwo_waypoint
            distance_to_waypoint = distancetwo_to_waypoint
            angle_err_to_waypoint = angletwo_err_to_waypoint

        
        print("\nGoing towards goal:\n")
        print(f"Distance to waypoint: {distance_to_waypoint}")
        print(f"Angle error to waypoint: {angle_err_to_waypoint}")
        print(f'vector to waypoint: {vector_waypoint}')

        goal_toleration = self.distance_tolerance-15

        # print(data)
        
        try:
            if not self.on_goal:
                # reset control flags to false except for update_robot
                for key in self.control_flags.keys():
                    if key != 'update_robot':
                        self.control_flags[key] = False

                if distance_to_waypoint > goal_toleration:

                    # Calculate the direction angle using the normalized orange_waypoint vector
                    direction_angle = math.degrees(math.atan2(
                        vector_waypoint[1], vector_waypoint[0]))

                    # Determine if the robot is moving straight or diagonally
                    if 45 <= abs(direction_angle) <= 135:
                        speed_scale = 1  # Scale factor for diagonal movements
                    else:
                        speed_scale = 0.5  # Scale factor for straight movements

                    # Calculate the speed using the distance to the waypoint
                    speed = self.pid_translation(
                        -distance_to_waypoint) * speed_scale
                    y = vector_waypoint[0] * speed
                    x = vector_waypoint[1] * speed

                    # set rotation to 0
                    rotation = 0
                    self.controller.publish_control_data(x, y, rotation)
                else:
                    self.on_goal = True

            elif abs(angle_err_to_waypoint) > self.angle_tolerance:
                rotation = -self.pid_rotation(angle_err_to_waypoint)
                print("rotation: ", rotation)
                # set x and y to 0
                x = y = 0

                self.controller.publish_control_data(x, y, rotation)
            else:
                print("Spitting out the balls")
                self.to_goal = False
                self.ball_out()
                self.stop_robot()

        except Exception as e:
            print(f"Error occurred in score_ball method: {e}")
            print('Stopping robot')
            self.stop_robot()


if __name__ == "__main__":
    queue = Queue(maxsize=10)
    manager = Manager()
    video_path = "/dev/video8"
    #video_path = 0

    broker_url = '192.168.1.101'
    broker_port = 1883
    topic = "robot/control"
    controller = Controller(broker_url, broker_port, topic)

    translation_pid = PID(Kp=0.02, Ki=0.000, Kd=0.001, setpoint=0)
    rotation_pid = PID(Kp=0.01, Ki=0.01, Kd=0.00, setpoint=0)

    translation_pid.output_limits = (0.25, 1)
    rotation_pid.output_limits = (-0.3, 0.3)

    control_flags = manager.dict({
        'update_orange_balls': False,
        'update_white_balls': False,
        'update_robot': True,
        'update_arena': False
    })

    control_logic = ControlLogic(
        queue, controller, translation_pid, rotation_pid, control_flags)

    # Suppress print statements in the camera_process
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

    camera_proc = Process(target=camera_process, args=(
        queue, video_path, control_flags))
    camera_proc.start()

    # Restore print statements in the main process
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    try:
        control_logic.run()
    except KeyboardInterrupt:
        camera_proc.terminate()
        camera_proc.join()
    finally:
        camera_proc.terminate()
        camera_proc.join()
