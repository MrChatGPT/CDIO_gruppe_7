#!/usr/bin/env pybricks-micropython

#https://pybricks.com/ev3-micropython/ev3devices.html#gyroscopic-sensor
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor
from pybricks.parameters import Port, Button, Stop
from pybricks.tools import wait
from pybricks.media.ev3dev import Font
from pybricks.media.ev3dev import ImageFile, Image


import time

from pybricks.tools import print, wait, StopWatch




ev3 = EV3Brick()

#Welcome greeting
#ev3.screen.load_image(ImageFile.EVIL)

# ev3.screen.load_image(ImageFile.EVIL)
# hello = "go fuck yourself"
ev3.screen.load_image("sponge2")
hello = "The names is Bond. James Bond"
ev3.speaker.say(hello)
wait(2000)
ev3.screen.clear()
    



#############################################################################
##Output C is for the right motor on the car, B is for the left motor.     ##
##A is for the medium/small motor                                          ##
##Uncomment all the places with motor0 when running the program on thursday##
#############################################################################

# Initialize a motor at port A. (Medium/Small)
motor0 = Motor(Port.A)

motor3 = Motor(Port.D)

# Initialize a motor at port B. (LARGE)
motor1 = Motor(Port.B)

# Initialize a motor at port C (LARGE)
motor2 = Motor(Port.C)

pressed_buttons = ev3.buttons.pressed()

# Play a sound.
#ev3.speaker.beep()

# # Run the motor up to 500 degrees per second. To a target angle of 90 degrees.
# motor1.run_target(500, 90)
# motor2.run_target(500, 90)


ev3.screen.print("UP - demo")
ev3.screen.print("DOWN - manuel")


# Keep doing nothing. 
while True:
    wait(1000) # 1 sec
    

    #Runs the Demo Program
    if Button.UP in ev3.buttons.pressed():
      ev3.screen.clear()
      ev3.screen.clear()
      ev3.screen.load_image(ImageFile.PINCHED_LEFT)
      wait(1000)
      ev3.screen.load_image(ImageFile.PINCHED_RIGHT)
      ev3.speaker.play_file('motor_start.wav')
      ev3.screen.load_image(ImageFile.PINCHED_MIDDLE)
      # Choose the "power" level for your wheels. Negative means reverse.
      motor0.dc(50) #remains constant speed
      motor3.dc(50) #remains constant speed
      #Drive forward
      motor1.dc(20)
      motor2.dc(20)

      wait(15000) # Wait for 15000 milliseconds = 15 seconds
      motor1.dc(50)
      motor2.dc(50)

      #Wait 10 sec before driving backwards
      wait(10000) # Wait for 10000 milliseconds = 10 seconds

      motor1.dc(-30)
      motor2.dc(-30)
      
      wait(5000) #Wait for 5000 milliseconds = 5 seconds
      
      ##########################!!!##############################
      #turn right?
      #ev3.screen.print("RIGHT")
      #Runs forward with a constant rotational speed of 500, and with an angle of 45 deg
      # motor1.run_angle(500,45)
      # motor2.run_angle(500,45)
      # motor1.dc(50)
      # motor2.dc(50)
    
      motor1.dc(50) #L
      motor2.dc(70) #R

     


      wait(5000) #Wait for 5000 milliseconds = 5 seconds
  
      ev3.screen.clear()
      #turn left?
      #ev3.screen.print("LEFT")
      # motor1.reset_angle(0)
      # motor2.reset_angle(0)

      # #Runs forward with a constant rotational speed of 500, and with an angle of 135 deg
      # motor1.run_angle(500,135)
      # motor2.run_angle(500,135)
      # motor1.dc(50)
      # motor2.dc(50)
      motor1.dc(70) #L
      motor2.dc(50) #R
      #ev3.screen.clear()

      wait(10000) # Wait for 10000 milliseconds = 10 seconds
      motor0.hold()
      motor3.hold()
      motor1.hold()
      motor2.hold()







    elif Button.LEFT in ev3.buttons.pressed():
        ev3.screen.clear()
        ev3.screen.print("Spongebob is... ")
        ev3.screen.print("So sweet <3")
       
    
    elif Button.RIGHT in ev3.buttons.pressed():
        ev3.screen.clear()
        ev3.screen.load_image(ImageFile.PINCHED_LEFT)
        wait(1000)
        ev3.screen.load_image(ImageFile.PINCHED_RIGHT)
        ev3.speaker.play_file('motor_start.wav')
        ev3.screen.load_image(ImageFile.PINCHED_MIDDLE)

    #Control the car manually
    elif Button.DOWN in ev3.buttons.pressed():
        ev3.screen.clear()
        ev3.screen.print("MANUEL")

 
     
       #Get it to turn left and right     
        while True:
            wait(1000)
            if Button.UP in ev3.buttons.pressed():
             motor0.dc(50)
             motor3.dc(50)
             motor1.dc(50)
             motor2.dc(50)

        

            if Button.DOWN in ev3.buttons.pressed():
              motor1.dc(-50)
              motor2.dc(-50)


            if Button.CENTER in ev3.buttons.pressed():
             # motor0.hold()
              ev3.screen.print("STOP")
              motor1.hold()
              motor2.hold()
              ev3.screen.clear()
            


            if Button.LEFT in ev3.buttons.pressed():
              ev3.screen.print("Turn Left")
              motor1.dc(70) #L
              motor2.dc(50) #R
              ev3.screen.clear()

            if Button.RIGHT in ev3.buttons.pressed():
              ev3.screen.print("Turn Right")
              motor1.dc(50) #L
              motor2.dc(70) #R
              ev3.screen.clear()

      
     
   
    
     #Only turns with an angle of 45 deg in the beginning
     #   motor1.run_angle(500,45, then=Stop.COAST, wait=False)
     #   motor2.run_angle(500,45, then=Stop.COAST, wait=False)
    
    #  #Runs with a constant rotational speed of 500, and with an angle of 45 deg
    #   motor1.run_angle(500,45)
    #   motor2.run_angle(500,45)

    #   #Run the DC motors at a constant duty cycle 50%
    #   motor1.dc(50)
    #   motor2.dc(50)
   

    #   if Button.RIGHT in ev3.buttons.pressed():
    #      motor0.hold()
    #      motor1.hold()
    #      motor2.hold()
     




 
