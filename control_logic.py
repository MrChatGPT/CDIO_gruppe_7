from multiprocessing import Process, Queue
from cam_module.cam2 import camera_process
import time
from queue import Empty  # Import Empty exception from the standard library queue module
from simple_pid import PID  # Import PID from simple-pid
from controller_module.controller import Controller
from time import sleep


class ControlLogic:
    def __init__(self, queue, controller, pid):
        self.queue = queue
        self.controller = controller
        self.pid = pid

    def run(self):
        while True:
            try:
                data = self.queue.get_nowait()
                if data is not None:
                    self.process_data(data)
                else:
                    self.stop_robot()
            except Empty:
                pass
            except Exception as e:
                print(f"Error occurred: {e}")
                break
            time.sleep(0.1)

    def process_data(self, data):
        #logic for waypoint:
        waypoint = data.get('vector_to_waypoint_robot_frame')
        distance_to_waypoint = data.get('distance_to_closest_waypoint')
       
        if waypoint is not None:
            print(f"Moving to waypoint: {waypoint}")
            print(f"Distance to waypoint: {distance_to_waypoint}")

            # Use the PID controller to get the speed
            speed = self.pid(-distance_to_waypoint)
            print(f"Speed: {speed}")

            # Scale the normalized vector to the speed
            y = waypoint[0] * speed
            x = waypoint[1] * speed
            rotation = 0
            print(f"Control data: x={x}, y={y}, rotation={rotation}")
            return

        #         self.controller.publish_control_data(x, y, rotation)
        #     else:
        #         self.stop_robot()

        # def stop_robot(self):
        #     # Stop the robot if no waypoint
        #     self.controller.publish_control_data(0, 0, 0)

        #----------------- Here we assume that the robot is on top of the waypoint ---------------------
        #logic for ball: 
        distance_to_ball = data.get('distance_to_closest_ball')
        ball = data.get('vector_to_ball_robot_frame')
        angle_err = data.get('angle_to_closest_ball')
        if ball is not None:
            print("Moving to ball: ", distance_to_ball)
            speed = self.pid(-distance_to_ball)
            print(f"Speed: {speed}")
            #Angle_err {0..180} & {-180..-0}
            if angle_err > 1:
                self.controller.publish_control_data(0,0,0.11)
                return
            elif angle_err < -1:
                self.controller.publish_control_data(0,0,-0.11)
                return
            if abs(angle_err) < 1:
                self.controller.publish_control_data(0,0.12,0)
                return
            if (distance_to_ball) < 160:
                self.controller.publish_control_data(0,0,0,1,0)
                #sæt evt. et dummy waypoint i et hjørne så vi forcer billedet til at blive opdateret (lappeløsning)....
                return
        
        print("\n\n\nVi burde aldrig se det her print weeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee\n\n\n")
            


            

            
            


if __name__ == "__main__":
    queue = Queue(maxsize=10)  # Set a reasonable max size for the queue
    video_path = "/home/madsr2d2/sem4/CDIO/CDIO_gruppe_7/camera2/film_2.mp4"

    # Initialize the Controller directly, which includes the MQTT client initialization
    broker_url = '192.168.1.101'
    broker_port = 1883
    topic = "robot/control"
    controller = Controller(broker_url, broker_port, topic)

    # Initialize the PID controller
    pid = PID(Kp=0.01, Ki=0.0, Kd=0.00, setpoint=0)
    pid.output_limits = (0, 1)  # Limit the PID output to the range of 0 to 1

    # Create an instance of the ControlLogic class
    control_logic = ControlLogic(queue, controller, pid)

    # Start the camera process
    camera_proc = Process(target=camera_process, args=(queue, video_path))
    camera_proc.start()

    # Run the control logic
    try:
        control_logic.run()
    except KeyboardInterrupt:
        camera_proc.terminate()
        camera_proc.join()
    finally:
        # Ensure the camera process is terminated when the control logic is done
        camera_proc.terminate()
        camera_proc.join()
