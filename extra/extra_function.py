def goal_draw(image, x, y):
 
    # Coordinates for the goal rectangle
    ##LEFT GOAL
    goal_x = x+20
    goal_y = y+430
    goal_w = 22
    goal_h = 130
  
    start_point_goal = (goal_x, goal_y)
    end_point_goal = (goal_x + goal_w, goal_y + goal_h)
    color_goal = (0, 255, 0)
    thickness_goal = 2
    
    image = cv2.rectangle(image, start_point_goal, end_point_goal, color_goal, thickness_goal)
    cv2.putText(image, 'L', (goal_x, goal_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_goal, thickness_goal)



    # Coordinates for the goal rectangle
    ##RIGHT GOAL
    goal_x = x+1360
    goal_y = y+470
    goal_w = 22
    goal_h = 75
    


    start_point_goal = (goal_x, goal_y)
    end_point_goal = (goal_x + goal_w, goal_y + goal_h)
    color_goal = (0, 255, 0)
    thickness_goal = 2
    
    image = cv2.rectangle(image, start_point_goal, end_point_goal, color_goal, thickness_goal)
    cv2.putText(image, 'R', (goal_x, goal_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_goal, thickness_goal)

    # # To display the image
    # cv2.imshow('Result', image)
    
    return image

def getImage():
    """This is just a dummy function. It will be replaced by the camera module."""
    
    # image = cv2.imread('test/images/WIN_20240403_10_40_59_Pro.jpg')
    # image = cv2.imread('test/images/WIN_20240403_10_39_46_Pro.jpg') 
    # image = cv2.imread('test/images/WIN_20240403_10_40_38_Pro.jpg') #hvid nej
    # image = cv2.imread('test/images/WIN_20240403_10_40_58_Pro.jpg') 
    # image = cv2.imread('test/images/pic50upsidedown.jpg') 
    # image = cv2.imread('test/images/WIN_20240410_10_31_43_Pro.jpg') 
    # image = cv2.imread('test/images/WIN_20240410_10_31_07_Pro.jpg') 
    # image = cv2.imread('test/images/WIN_20240410_10_31_07_Pro.jpg') #orig pic with transfrom new
    # image = cv2.imread('test/images/pic50egghorizontal.jpg') 
    # image = cv2.imread('test/images/WIN_20240410_10_30_54_Pro.jpg') 
    image = cv2.imread('test/images/WIN_20240610_09_33_12_Pro.jpg') 
    
    return image

#Used for the cross (and arena), but not limited to
def square_draw(image, x, y, w, h, area):
    # # Start coordinate, here (x, y)
    # start_point = (x+60, y)
    
    # # End coordinate
    # end_point = (x+60,y+h)
    
    # # Green color in BGR
    # color = (0, 255, 0)  # Using a standard green color; modify as needed
    
    # # Line thickness of 2 px
    # thickness = 2
    
    # # Using cv2.rectangle() method to draw a rectangle around the car
    # image = cv2.line(image, start_point, end_point, color, thickness)
    
    # # Optionally, add text label if needed
    # cv2.putText(image, 'cross', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    

    # image = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2) 


    # print("before recreating the points")

     # Recreate the rectangle points from x, y, w, h
    rect_points = np.array([
        [x, y],
        [x + w, y],
        [x, y + h],
        [x + w, y + h]
    ], dtype=np.float32)

    # print(f"rect_points={rect_points}")
    # print("after recreating the points")
    # The points need to be ordered correctly for minAreaRect to work
    rect_points = cv2.convexHull(rect_points)

    # Find the minimum area rectangle
    min_area_rect = cv2.minAreaRect(rect_points)

    """ Cross
    minimum area rectangle=((1100.0, 523.5), (167.0, 168.0), 90.0)
    The first two numbers (1100.0, 523.5), shows the x and y-axis of the central point in the square.
    """
    # print(f"minimum area rectangle={min_area_rect}") 

    # Convert the rectangle to box points (four corners)
    """
    Box prints out the the coordinates respectively:
    box=[xTopleft,yTopleft],
        [xTopright, yTopright],
        [xBottomright, yBottomright],
        [xBottomleft,yBottomleft]
    """
    box = cv2.boxPoints(min_area_rect)
    # print(f"box={box}")
    box = np.int0(box)
 



    return box, min_area_rect

def line_draw(image, x, y, w, h, area):

    # Green color in BGR 
    color = (0, 255, 0) 
    
    # Line thickness of 9 px 
    thickness = 9
 


    # represents the top left corner of image 
    start_point = (x, y) 
    # represents the top right corner of image 
    end_point = (x+w, y) 
    # Draw a diagonal green line with thickness of 9 px 
    image = cv2.line(image, start_point, end_point, color, thickness) 




    # represents the top left corner of image 
    start_point = (x, y) 
    # represents the bottom left corner of image 
    end_point = (x, y+h) 
    # Draw a diagonal green line
    image = cv2.line(image, start_point, end_point, color, thickness) 



    # represents the top right corner of image 
    start_point = (x+w, y) 
    # represents the bottom right corner of image 
    end_point = (x+w, y+h) 
    # Draw a diagonal green line
    image = cv2.line(image, start_point, end_point, color, thickness) 

    # represents the bottom left corner of image 
    start_point = (x, y+h) 
    # represents the bottom right corner of image 
    end_point = (x+w, y+h) 
    # Draw a diagonal green line
    image = cv2.line(image, start_point, end_point, color, thickness) 

    return image

def car_draw(image, x, y, w, h, area):
    # Start coordinate, here (x, y), represents the top left corner of rectangle 
    start_point = (x, y)
    
    # End coordinate, here (x+w, y+h), represents the bottom right corner of rectangle
    end_point = (x+w, y+h)
    
    # Green color in BGR
    color = (0, 255, 0)  # Using a standard green color; modify as needed
    
    # Line thickness of 2 px
    thickness = 2
    
    # Using cv2.rectangle() method to draw a rectangle around the car
    image = cv2.rectangle(image, start_point, end_point, color, thickness)
    
    # Optionally, add text label if needed
    cv2.putText(image, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    
    return image

def CannyEdgeGray(image):
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
   cv2.imshow("gray pic", gray) 
   gray = cv2.bilateralFilter(gray, 11, 17, 17)
   cv2.imshow("gray bilateral", gray) 
 #    cv2.imshow("Gray image", gray) 
   gray = cv2.GaussianBlur(gray, (5, 5), 0)
 #    cv2.imshow("Gaussian Blur", gray) 

   edged = cv2.Canny(gray,0, 105)
   cv2.imshow("Canny edge B/W detection", edged) 

   #cropped_image = edged[240:140, 168:167] # Slicing to crop the image

   # Display the cropped image