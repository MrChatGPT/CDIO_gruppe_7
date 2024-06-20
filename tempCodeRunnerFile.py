a_handler = initialize()

try:
    while True:
        process_frame(camera_handler)
finally:
    # Ensure the camera is released properly
    camera_handler.release_camera()
    comstop = (0, 0, 0, 0, 0)
    publish_c