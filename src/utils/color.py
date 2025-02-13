from PIL import Image, ImageEnhance
import cv2
import numpy as np

target_light = 147.15994736012124 # target lightness using 724 image.
def white_balance(image):
    result = cv2.xphoto.createSimpleWB().balanceWhite(image)
    return result

def get_pixel_value(image, event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel_value = image[y,x]
        print(f'Pixel value at ({x}, {y}): {pixel_value}')

def equalize(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:,:,2]= cv2.equalizeHist(hsv[:,:,2])
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return result

image = cv2.imread('data/raw_images/724_LDLeanColour.JPG')
half = cv2.resize(image, (0,0), fx=0.2, fy=0.2)
balance = white_balance(half)
lab = cv2.cvtColor(balance, cv2.COLOR_BGR2LAB)
l_channel, a_channel, b_channel = cv2.split(lab)
print(np.mean(l_channel))
cv2.imshow("Image", lab)
cv2.waitKey()
cv2.destroyAllWindows

    
    
