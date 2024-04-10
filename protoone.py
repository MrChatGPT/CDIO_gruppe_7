import socket
from pynput import keyboard

# Konfigurer UDP
udp_ip = "192.168.1.210"
udp_port = 8888 # Portnummer
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP

# Mapping of keys to characters
mapping = {
    keyboard.Key.up: "A",
    keyboard.Key.down: "B",
    keyboard.Key.left: "L",
    keyboard.Key.right: "R"
}tisseamnd

# Funktion til at sende en besked via UDP
def send_udp_message(message):
    sock.sendto(message.encode(), (udp_ip, udp_port))
    print(f"Sent: {message}")

# Definerer en funktion til at håndtere input
def handle_input(key):
    if key in mapping:
        send_udp_message(mapping[key])
    else:
        print(f"No mapping found for: {key}")

# Lytter efter piletaster og deres kombinationer
def on_press(key):
    try:
        if key.char is not None:
            handle_input(key.char)
    except AttributeError:
        if key in mapping:
            handle_input(key)

# Kør programmet
if __name__ == "__main__":
    print("Press arrow keys or their combinations. Press CTRL+C to stop.")
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
