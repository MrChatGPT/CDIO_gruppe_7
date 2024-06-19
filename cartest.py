from simple_pid import PID
import time

def main():
    # Setpoint is 10 as we want the output to be 1 when distance is 1000
    pid = PID(1, 0.0001, 0.001)  # Starting with small PID coefficients
    pid.setpoint = 0
    pid.output_limits = (-1000, 1000)
    
    distance = 1200
    while True:
        pid_output = pid(distance)  # Divide by 1000 to normalize distance
        distance = distance + pid_output / 1000
        print(f"Distance: {distance}, PID Output: {pid_output/1000}")
        time.sleep(0.01)

if __name__ == "__main__":
    main()
