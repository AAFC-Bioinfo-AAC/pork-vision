from PIL import Image, ImageEnhance
import cv2
import numpy as np
from skimage.exposure import match_histograms,equalize_hist

def white_balance(image):
    result = cv2.xphoto.createSimpleWB().balanceWhite(image)
    return result

def get_pixel_value(event, x, y, flag, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel_value = params[y,x]
        print(f'Pixel value at ({x}, {y}): {pixel_value}')

def equalize(image):
    '''
    ISSUE
    '''
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:,:,2]= cv2.equalizeHist(hsv[:,:,2])
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return result

def adjust_brightness(image, target_mean_lightness, target_mean_a, target_mean_b):
    '''
    DEPRECATED
    '''
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    current_mean_lightness = np.mean(l_channel)
    current_mean_a = np.mean(a_channel)
    current_mean_b = np.mean(b_channel)
    #print(f"Current mean lightness: {current_mean_lightness}")
    #print(f"Target mean lightness: {target_mean_lightness}")
    scale_l = target_mean_lightness/current_mean_lightness
    scale_a = target_mean_a/current_mean_a
    scale_b = target_mean_b/current_mean_b
    l_channel = np.clip(l_channel*scale_l,0,255).astype(np.uint8)
    a_channel = np.clip(a_channel*scale_a,0,255).astype(np.uint8)
    b_channel = np.clip(b_channel*scale_b,0,255).astype(np.uint8)
    #print(f"Adjusted lightness to {np.mean(l_channel)}")
    adjusted_lab_image = cv2.merge([l_channel, a_channel, b_channel])
    adjusted_image = cv2.cvtColor(adjusted_lab_image, cv2.COLOR_LAB2BGR)
    return adjusted_image

def reference_standardize(images, reference_image):
    standardized_images = []
    reference_image = cv2.fastNlMeansDenoisingColored(reference_image,None,3,3,7,21)
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        standard_img = match_histograms(img_rgb, reference_image, channel_axis=-1)
        #standard_img = cv2.medianBlur(img, 3)
        standard_img = cv2.cvtColor(standard_img, cv2.COLOR_RGB2BGR)
        #standard_img = cv2.fastNlMeansDenoisingColored(img,None,3,3,7,21)
        standardized_images.append(standard_img)
    return standardized_images

reference_image = cv2.imread('data/raw_images/724_LDLeanColour.JPG')
reference_image= cv2.resize(reference_image, (0,0), fx=0.15, fy=0.15)
reference_image = white_balance(reference_image)
reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
img_list = ['data/raw_images/724_LDLeanColour.JPG','data/raw_images/1704_LdLeanColor.JPG', 'data/raw_images/1701_LdLeanColor.JPG', 'data/raw_images/2401_LdLeanColor.JPG']
ready_imgs = []

for img in img_list:
    print(img)
    image = cv2.imread(img)
    half = cv2.resize(image, (0,0), fx=0.15, fy=0.15)
    #### For testing purposes
    cv2.imshow("Image", half)
    cv2.setMouseCallback("Image", get_pixel_value, half)
    cv2.waitKey()
    cv2.destroyAllWindows()

    balance = white_balance(half)
    ready_imgs.append(balance)
standardized = reference_standardize(ready_imgs, reference_image)

for img in standardized:
    cv2.imshow("Standard", img)
    cv2.setMouseCallback("Standard", get_pixel_value, img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    
    
