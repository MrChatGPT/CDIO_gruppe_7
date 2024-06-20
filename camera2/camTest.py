import cv2


def test_webcam():
    # Open a connection to the webcam using the V4L2 backend
    # camPath = "/dev/video9"
    path = "/dev/video8"
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to exit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print(
                "Error: Failed to capture image. Check if the webcam is connected and accessible.")
            break

        # Display the resulting frame
        cv2.imshow('Webcam Test', frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture and close the window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_webcam()
