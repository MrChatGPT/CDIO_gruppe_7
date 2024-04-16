from pynput import keyboard

def on_press(key):
    try:
        print(f'Key {key} pressed')
    except AttributeError:
        print('Special key {key} pressed')

def on_release(key):
    print(f'Key {key} released')
    if key == keyboard.Key.esc:
        # Stop listener
        return False

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()

# import socket
# from pynput import keyboard

# # Configure UDP
# udp_ip = "192.168.204.36"
# udp_port = 8888  # Port number
# sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP

# pressed_keys = set()

# def check_key_combinations():
#     if len(pressed_keys) < 1:
#         print("No keys pressed")
#     elif len(pressed_keys) == 1:
#         print("one key pressed")
#         if keyboard.Key.up in pressed_keys:
#             print("moving farwards")
  
#     if keyboard.Key.up in pressed_keys and keyboard.Key.right in pressed_keys:
#         send_udp_message("Moving Up and Right")
#     # Add more combinations as needed

# # Listener for key presses
# def on_press(key):
#     pressed_keys.add(key)
#     check_key_combinations()

# # Listener for key releases
# def on_release(key):
#     pressed_keys.discard(key)




# # Function to send a message via UDP
# def send_udp_message(message):
#     sock.sendto(message.encode(), (udp_ip, udp_port))
#     print(f"Sent: {message}")


# if __name__ == "__main__":
#     print("Press arrow keys or their combinations. Press CTRL+C to stop.")
#     with keyboard.Listener(on_press=on_press) as listener:
#         listener.join()
