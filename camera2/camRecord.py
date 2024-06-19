import cv2


def test_webcam():
    # Open a connection to the webcam using the V4L2 backend
    camPath = "/dev/video9"
    cap = cv2.VideoCapture(camPath, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec
    # Output file, codec, FPS, frame size
    out = cv2.VideoWriter('output.avi', fourcc, 30.0, (640, 480))

    print("Press 'q' to exit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print(
                "Error: Failed to capture image. Check if the webcam is connected and accessible.")
            break

        # Write the frame to the video file
        out.write(frame)

        # Display the resulting frame
        cv2.imshow('Webcam Test', frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture, video writer, and close the window
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_webcam()
