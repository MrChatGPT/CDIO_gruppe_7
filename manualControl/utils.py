
import time
import threading
import sys
from pyPS4Controller.controller import Controller


class MyController(Controller):
    def __init__(self, interface="/dev/input/js0", connecting_using_ds4drv=False, **kwargs):
        super().__init__(interface=interface,
                         connecting_using_ds4drv=connecting_using_ds4drv, **kwargs)
        self.R3_value = [0, 0]
        self.L3_value = [0, 0]
        self.R1_value = 0
        self.x_value = 0

    # Function is called when R3 is moved
    def on_R3_x_at_rest(self):
        self.R3_value[0] = 0

    def on_R3_y_at_rest(self):
        self.R3_value[1] = 0

    def on_R3_down(self, value):
        self.R3_value[1] = self.map_stick_value(value)

    def on_R3_up(self, value):
        self.R3_value[1] = self.map_stick_value(value)

    def on_R3_left(self, value):
        self.R3_value[0] = self.map_stick_value(value)

    def on_R3_right(self, value):
        self.R3_value[0] = self.map_stick_value(value)

    # Function is called when L3 is moved
    def on_L3_x_at_rest(self):
        self.L3_value[0] = 0

    def on_L3_y_at_rest(self):
        self.L3_value[1] = 0

    def on_L3_down(self, value):
        self.L3_value[1] = self.map_stick_value(value)

    def on_L3_up(self, value):
        self.L3_value[1] = self.map_stick_value(value)

    def on_L3_left(self, value):
        self.L3_value[0] = self.map_stick_value(value)

    def on_L3_right(self, value):
        self.L3_value[0] = self.map_stick_value(value)

    # Function is called when R1 is pressed
    def on_R1_press(self):
        self.R1_value = 1

    def on_R1_release(self):
        self.R1_value = 0

    # Function is called when X is pressed
    def on_x_press(self):
        self.x_value = 1

    def on_x_release(self):
        self.x_value = 0

    def start(self):
        self.listen_thread = threading.Thread(
            target=self.listen, args=(60,), daemon=True)
        self.listen_thread.start()

    @staticmethod
    def map_stick_value(raw_value):
        return raw_value / 32767.0


# Main loop to display the R3 value, exiting after 60 seconds for example
# Main loop to display the R3 value, exiting after 60 seconds for example
if __name__ == "__main__":
    controller = MyController()  # Create the controller object
    controller.start()  # Start the controller thread

    start_time = time.time()
    while time.time() - start_time < 10:  # Run for 10 seconds
        print(f"R3: {controller.R3_value}, L3: {controller.L3_value}, R1: {controller.R1_value}, x: {controller.x_value}", flush=True)
        time.sleep(0.1)  # Delay to prevent flooding the output

    sys.exit(0)
