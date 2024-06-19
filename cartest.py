from simple_pid import PID
import time
pid_forwards = PID(0.5, 0.0001, 0.001, setpoint=0)  # Starting with small PID coefficients
pid_forwards.output_limits = (-1000, 1000)

pid_turn = PID(0.2, 0.01, 0.001, setpoint=0)  # Starting with small PID coefficients
pid_turn.output_limits = (-180, 180) #180*0.88

def main():
    if abs(angle_error) > 0.5:
        pid_output = - pid_turn(angle_error) / 250 # now it should go between -0.72 and +0.72
        if pid_output < 0:
            pid_output = (pid_output - 0.12) # at most -0.84
        elif pid_output > 0:
            pid_output = (pid_output + 0.12)
        if angle_error < 5:
            publish_controller_data((0, 0, pid_output, 0, 0))
            time.sleep(0.05)
            publish_controller_data((0, 0, pid_output, 0, 0))

        publish_controller_data((0, 0, pid_output, 0, 0))
        continue
    forwards = - pid_forwards(distance)/1500 # max 0.66
    forwards = forwards + 0.12
    if distance < 50:
        publish_controller_data((0, forwards, 0, 0, 0))
        time.sleep(0.05)
        publish_controller_data((0, 0, 0, 0, 0))

      
