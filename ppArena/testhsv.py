#https://github.com/alieldinayman/hsv-color-picker/blob/master/HSV%20Color%20Picker.py

import cv2
import numpy as np
import sys
import tkinter as tk
from tkinter import filedialog

import pandas as pd
import matplotlib.pyplot as plt


  # Lists to store the HSV values
lower_hsv_values = []
upper_hsv_values = []

# Global variables
df_lower = None
df_upper = None

image_hsv = None
pixel = (0,0,0) #RANDOM DEFAULT VALUE

ftypes = [
    ("JPG", "*.jpg;*.JPG;*.JPEG"), 
    ("PNG", "*.png;*.PNG"),
    ("GIF", "*.gif;*.GIF"),
    ("All files", "*.*")
]

def check_boundaries(value, tolerance, ranges, upper_or_lower):
    if ranges == 0:
        # set the boundary for hue
        boundary = 180
    elif ranges == 1:
        # set the boundary for saturation and value
        boundary = 255

    if(value + tolerance > boundary):
        value = boundary
    elif (value - tolerance < 0):
        value = 0
    else:
        if upper_or_lower == 1:
            value = value + tolerance
        else:
            value = value - tolerance
    return value

def pick_color(event,x,y,flags,param):
    global df_lower, df_upper  # Declare df_lower and df_upper as global

    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = image_hsv[y,x]

        #HUE, SATURATION, AND VALUE (BRIGHTNESS) RANGES. TOLERANCE COULD BE ADJUSTED.
        # Set range = 0 for hue and range = 1 for saturation and brightness
        # set upper_or_lower = 1 for upper and upper_or_lower = 0 for lower
        hue_upper = check_boundaries(pixel[0], 10, 0, 1)
        hue_lower = check_boundaries(pixel[0], 10, 0, 0)
        saturation_upper = check_boundaries(pixel[1], 10, 1, 1)
        saturation_lower = check_boundaries(pixel[1], 10, 1, 0)
        value_upper = check_boundaries(pixel[2], 40, 1, 1)
        value_lower = check_boundaries(pixel[2], 40, 1, 0)




        upper =  np.array([hue_upper, saturation_upper, value_upper])
        lower =  np.array([hue_lower, saturation_lower, value_lower])



     ######Create dataframe#######

     
   

        # Create a temporary list for the current lower and upper HSV values
        temp_lower = [hue_lower, saturation_lower, value_lower]
        temp_upper = [hue_upper, saturation_upper, value_upper]

        # Append to the main list
        lower_hsv_values.append(temp_lower)
        upper_hsv_values.append(temp_upper)

        # At the end, create DataFrames from the lists
        df_lower = pd.DataFrame(lower_hsv_values, columns=['Hue', 'Saturation', 'Value'])
        df_upper = pd.DataFrame(upper_hsv_values, columns=['Hue', 'Saturation', 'Value'])






        #print(lower , upper)
        print("lower" + str(lower) + " upper" + str(upper))



        #A MONOCHROME MASK FOR GETTING A BETTER VISION OVER THE COLORS 
        image_mask = cv2.inRange(image_hsv,lower,upper)
        cv2.imshow("Mask",image_mask)


def avgmean():

   global df_lower, df_upper  # Declare df_lower and df_upper as global

   average_lower = df_lower.mean()
   median_lower = df_lower.median()

   average_upper = df_upper.mean()
   median_upper = df_upper.median()
   
  #To print the averages and medians
   print("Average Lower HSV:\n", average_lower)
   print("Median Lower HSV:\n", median_lower)
   print("Average Upper HSV:\n", average_upper)
   print("Median Upper HSV:\n", median_upper)

   print("Lower")
   print(df_lower.describe())
   print("Upper")
   print(df_upper.describe())
   
  # Assuming your DataFrames are named df_lower and df_upper 
  # Plotting boxplots for df_lower
   plt.figure(figsize=(10, 6))
   plt.subplot(1, 2, 1)  # First subplot for the 'lower' values
   df_lower.boxplot()
   plt.title('Lower HSV Values')

   plt.subplot(1, 2, 2)  # Second subplot for the 'upper' values
   df_upper.boxplot()
   plt.title('Upper HSV Values')

   plt.tight_layout()
   plt.show()



def main():

    global image_hsv, pixel

    #OPEN DIALOG FOR READING THE IMAGE FILE
    # root = tk.Tk()
    # root.withdraw() #HIDE THE TKINTER GUI
    # file_path = filedialog.askopenfilename(filetypes = ftypes)
    # root.update()
    #image_src = cv2.imread(file_path)
    #image_src = cv2.imread('/home/slothie/CDIO_gruppe_7/ppArena/test/images/WIN_20240403_10_40_58_Pro.jpg') 
    image_src = cv2.imread('/home/slothie/CDIO_gruppe_7/ppArena/test/images/WIN_20240403_10_40_38_Pro.jpg') #hvid nej
    cv2.imshow("BGR",image_src)

    #CREATE THE HSV FROM THE BGR IMAGE
    image_hsv = cv2.cvtColor(image_src,cv2.COLOR_BGR2HSV)
    cv2.imshow("HSV",image_hsv)

    #CALLBACK FUNCTION
    cv2.setMouseCallback("HSV", pick_color)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

    avgmean()

if __name__=='__main__':
    main()