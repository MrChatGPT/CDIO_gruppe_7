# hello from me
import time
import threading
from pyPS4Controller.controller import Controller


# def map_stick_value(raw_value):
#     return raw_value / 32767.0


# class MyController(Controller):

#     def __init__(self, **kwargs):
#         self.R3_value = 0
#         super().__init__(**kwargs)

#     def on_R3_x_at_rest(self):
#         self.R3_value = 0

#     def on_R3_y_at_rest(self):
#         self.R3_value = 0

#     def on_R3_down(self, value):
#         self.R3_value = map_stick_value(value)

#     def on_R3_up(self, value):
#         self.R3_value = map_stick_value(value)

#     def on_R3_left(self, value):
#         self.R3_value = map_stick_value(value)

#     def on_R3_right(self, value):
#         self.R3_value = map_stick_value(value)


# controller = MyController(interface="/dev/input/js0",
#                           connecting_using_ds4drv=False)
# # you can start listening before controller is paired, as long as you pair it within the timeout window
# controller.listen(timeout=60)

# while True:
#     print(controller.R3_value)


# # hello from me


def map_stick_value(raw_value):
    return raw_value / 32767.0


class MyController(Controller):
    def __init__(self, **kwargs):
        self.R3_value = [0, 0]
        super().__init__(**kwargs)

    def on_R3_x_at_rest(self):
        self.R3_value[0] = 0

    def on_R3_y_at_rest(self):
        self.R3_value[1] = 0

    def on_R3_down(self, value):
        self.R3_value[1] = map_stick_value(value)

    def on_R3_up(self, value):
        self.R3_value[1] = map_stick_value(value)

    def on_R3_left(self, value):
        self.R3_value[0] = map_stick_value(value)

    def on_R3_right(self, value):
        self.R3_value[0] = map_stick_value(value)


# Global controller instance
controller = MyController(interface="/dev/input/js0",
                          connecting_using_ds4drv=False)


def listen_to_controller():
    controller.listen(timeout=60)


# Set up threading for the controller listening
controller_thread = threading.Thread(target=listen_to_controller)
controller_thread.start()

# Main loop to display the R3 value, exiting after 60 seconds for example
start_time = time.time()
while time.time() - start_time < 60:
    print(f"Current R3 value: {controller.R3_value}", flush=True)
    time.sleep(0.1)  # Delay to prevent flooding the output

# Optionally wait for the controller thread to finish
controller_thread.join()
