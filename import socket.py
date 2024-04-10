import socket

# EV3 settings
server_ip = '192.168.1.68'  # Replace with the actual IP address of your EV3
server_port = 7777          # Make sure this matches the port in the EV3's server script

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

try:
    while True:
        message = input("Enter message to send to EV3: ")
        if message.lower() == "exit":
            break

        # Send data
        sock.sendto(message.encode(), (server_ip, server_port))
        print(f"Sent '{message}' to EV3")
finally:
    print("Closing socket")
    sock.close()
