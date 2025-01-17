# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 15:45:08 2025

@author: jivdn
"""

#all imports
import cv2
import numpy as np
import struct
import snap7
from matplotlib import pyplot as plt

PLC_PORT = 2001  
plc = snap7.client.Client() 
plc.connect('192.168.0.51', 0, 1)  

#for communication
def write_lreal_db(db_number, start_address, value): 
    plc.db_write(db_number, start_address, bytearray(struct.pack('>d', value)))  
    return None

def write_bool(db_number, start_offset, bit_offset, value):  # To write 1 bit to a specific variable in a DB
    reading = plc.db_read(db_number, start_offset, 1)  # (db number, start offset, read 1 byte)
    snap7.util.set_bool(reading, 0, bit_offset, value)  # (value 1= true;0=false) (bytearray_: bytearray, byte_index: int, bool_index: int, value: bool)
    plc.db_write(db_number, start_offset, reading)  # write back the bytearray and now the boolean value is changed in the PLC.
    return None

def read_bool(db_number, start_offset, bit_offset):  
    reading = plc.db_read(db_number, start_offset, 1)
    a = snap7.util.get_bool(reading, 0, bit_offset)
    return a

#camera on
cv2. WINDOW_AUTOSIZE
cv2.namedWindow("webcam", cv2.WINDOW_AUTOSIZE)

cv2.moveWindow("webcam",100,000) 
webcam = cv2.VideoCapture(1) #1 werkt voor usb port, 0 voor camera in laptop

webcam.set(cv2.CAP_PROP_FRAME_WIDTH,1920)  #dimensions (don't change it!)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)


while True:
    retval, webcam_img = webcam.read()
    if(retval == True):         
        cv2.imshow("webcam",webcam_img) 
        
        key = cv2.waitKey(10)
        if key == ord('s'):
            cv2.imwrite("webcam_img.jpg", webcam_img)
            
            img_gray = cv2.cvtColor(webcam_img, cv2.COLOR_BGR2GRAY)   #grey image
            blurred_img = cv2.GaussianBlur(webcam_img, (21, 21), 30)  #used to reduce noise in image
            edges = cv2.Canny(img_gray, 40, 140)                      #edge detection
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  #finds contours
            
            mask = np.zeros_like(edges)
            max_area = 0 #starting 
            rightmost_pixel = (0, 0)  # To track the coordinates of the rightmost pixel
            center = (0, 0)  
            shape_centers = []  
            
            for cnt in contours:
                epsilon = 0.05 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                number_of_vertices = len(approx)
                if number_of_vertices == 3 or number_of_vertices == 4:  #3 for triangles 4 for squares
                    cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

            
                    area = cv2.contourArea(cnt)#area
                    if area > max_area:
                        max_area = area  # Update the maximum area
                     
                        # Calculate the center 
                        M = cv2.moments(cnt)
                        if M['m00'] != 0:  
                            centroid = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

                  
                        rightmost_pixel = tuple(cnt[cnt[:, :, 0].argmax()][0])  # Rightmost point of the contour

                        #midden van bloje bepalen
                        x, y, w, h = cv2.boundingRect(cnt)
                        center = (x + w // 2, y + h // 2)
                        shape_centers.append(center)
                        
            # Calculate the line between the centroid and the rightmost pixel
            dx = abs(rightmost_pixel[0] - centroid[0])  
            dy = abs(rightmost_pixel[1] - centroid[1])  
            line_length = int(np.sqrt(dx ** 2 + dy ** 2))  #pythagoras
            
            
            # print("Line Length:", line_length ,"pixels")
            # print("X:", dx, "pixels")
            # print("Y:",dy, "pixels")
            dx_length_in_mm= dx/4
            dy_length_in_mm= dy/4 #4 pixels per mm
            print("Rightmost pixel coordinates:", rightmost_pixel)
            #
            meest_rechter_pixel_x=(rightmost_pixel[0])
            meest_rechter_pixel_y=(rightmost_pixel[1])
            
            vaste_coordinaat_van_PLC_x= -4.47
            vaste_coordinaat_van_PLC_y= -40.0
            
            x_coordinaat_voor_meest_rechter_pixel= ((meest_rechter_pixel_x/4.0)*-1)+ vaste_coordinaat_van_PLC_x#hoeveelheid mm + coordinaat robot
            y_coordinaat_voor_meest_rechter_pixel= (meest_rechter_pixel_y/4.0)+ vaste_coordinaat_van_PLC_y
            
            x_coordinaat_van_midden= x_coordinaat_voor_meest_rechter_pixel + dx_length_in_mm
            y_coordinaat_van_midden=y_coordinaat_voor_meest_rechter_pixel + dy_length_in_mm
            
            write_lreal_db(22, 16, x_coordinaat_van_midden )#verstuuren pixel (misschien x en y nog omdraaien)
            write_lreal_db(22, 24, y_coordinaat_van_midden )
            print("x_coordinaat_van_midden", x_coordinaat_van_midden)
            print("y_coordinaat_van_midden", y_coordinaat_van_midden)
            
          
            background_mask = cv2.bitwise_not(mask)
            background = cv2.bitwise_and(blurred_img, blurred_img, mask=background_mask)


            shapes = cv2.bitwise_and(webcam_img, mask=mask)
            final_img = cv2.add(background, shapes)

            output_img = final_img.copy()
            for cnt in contours:
                epsilon = 0.05 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                num_vertices = len(approx)
                shape_name = ""
                color = (0, 0, 0) #zwart.

              
                area = cv2.contourArea(cnt)# area uitgedrukt in pixels

          
                if area < 100: #ruis in achterdrond vermijden en alleen de grootste vorm nemen
                    continue

            shape_name = "Square"
            if 92000 <= area <= 92500: #hoeveel hied pixels die gedetecteerd kunnen worden
                color = (0,255,0)  
                print("square detected")
                write_bool(22, 2, 0, 1) #offset nog aanpassen
                
        elif num_vertices == 3:  # Triangle
            shape_name = "Triangle"
            if 180000 <= area <= 190000:
                color = (0,255,0) 
                print("orange or green triangle")
                write_bool(22, 2, 0, 1) #
                
            elif 45000 <= area <= 47000:
                color = (0,255,0) 
                print("white or dark blue triangle")
                write_bool(22, 2, 0, 1) #
                
            elif 91000 <= area <= 93000:
                color = (0,255,0) 
                print("red triangle")
                write_bool(22, 2, 0, 1)#
            else:
                continue 

    
                cv2.drawContours(output_img, [approx], -1, color, 2)
                x, y, w, h = cv2.boundingRect(approx)
            
                cv2.putText(output_img, f"{shape_name}: {area:.2f} px", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            print(f"Maximum area found: {max_area:.2f} pixels")

            
            cv2.line(output_img, centroid, rightmost_pixel, (0, 255, 0, 2)) 
            cv2.line(output_img, centroid, (rightmost_pixel[0], centroid[1]), (0, 255,0) )
            cv2.line(output_img, centroid, (centroid[0], rightmost_pixel[1]), (0,255,0), 2) 

          
            for center in shape_centers:
                cv2.circle(output_img, center, 10, (0, 255, 0), -1)  #groene punt in het midden aantonen

   
            cv2.imshow("Detected Shapes with Lines", output_img)# foto met lijnen

         
            plt.subplot(121), plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
            plt.title('Blurred Background'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
            plt.title('Detected Shapes with Lines'), plt.xticks([]), plt.yticks([])

            plt.show()

        elif key == 27:  # Exit on ESC
            break
    else:
        print("Failed to read from webcam. Exiting.")
        break

webcam.release()
cv2.destroyAllWindows()

                        

   
