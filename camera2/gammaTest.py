import cv2
import numpy as np


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) *
                     255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def read_and_display_video(video_path, apply_gamma_correction=False, gamma_value=2.2):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Unable to open the video file {video_path}.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Optional: Apply gamma correction
        if apply_gamma_correction:
            frame = adjust_gamma(frame, gamma=gamma_value)

        # Display the frame
        cv2.imshow('Video Playback', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = '/home/madsr2d2/sem4/CDIO/CDIO_gruppe_7/camera2/output1.avi'

    # Without gamma correction
    read_and_display_video(video_path, apply_gamma_correction=False)

    # With gamma correction
    # read_and_display_video(video_path, apply_gamma_correction=True, gamma_value=2.2)
