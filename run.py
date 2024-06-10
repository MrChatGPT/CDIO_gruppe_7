from matplotlib import pyplot as plt 
from picture.livefeed import *
from picture.transform_arena import *
from picture.utils import *

def basicDetectofImage(image):
    circle_detection(image)  #THIS IS THE GOOD SHIT
    detect_ball_colors(image)

#Used for detecting objects in picture, and the colors in the image (with image correction, (Transform))
def getMeSomeBallInfo(image):
    # image = perspectiveTrans(image) #Problems when cutting edges off, making the correct size wxh
    image, circles_info = circle_detection(image)  # THIS IS THE GOOD SHIT
    # image, orange_detected = detect_ball_colors(image)
    image = detect_ball_colors(image)
    # Print stored circles information
    print("Detected and Stored Circles:")
    for circle in circles_info:
        print(circle)
 
def transform_and_detect(image):
    # calibrate(image)
    image = transform(image)
    circle_detection(image) 
    image = detect_ball_colors(image)

def getVideo():
    stream_url = 'http://172.28.32.1:8080'

    # Capturing video through webcam 
    webcam = cv2.VideoCapture(stream_url)

    # Start a while loop 
    while True:
        # Reading the video from the webcam in image frames 
        ret, imageFrame = webcam.read()
        
        if not ret:
            print("Failed to capture image")
            break

        transform_and_detect(imageFrame)

        # Display the resulting frame
        cv2.imshow("Multiple Color Detection in Real-Time", imageFrame)

        # Break the loop on 'q' key press
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release the webcam and destroy all OpenCV windows
    webcam.release()
    cv2.destroyAllWindows()


camera_handler = CameraHandler()

# Start video in a separate thread
camera_handler.start_video()
i = 0
try:
    while True:
        image = camera_handler._run_video()
        transform_and_detect(image)
        
        
finally:
    # Ensure the camera is released properly
    camera_handler.release_camera()
