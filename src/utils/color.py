from PIL import Image, ImageEnhance
import cv2
import numpy as np

def white_balance(image):
    result = cv2.xphoto.createSimpleWB().balanceWhite(image)
    return result

def get_pixel_value(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel_value = image[y,x]
        print(f'Pixel value at ({x}, {y}): {pixel_value}')

def equalize(image):
    '''
    ISSUE
    '''
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:,:,2]= cv2.equalizeHist(hsv[:,:,2])
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return result

def adjust_brightness(image, target_mean_lightness):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    current_mean_lightness = np.mean(l_channel)
    print(f"Current mean lightness: {current_mean_lightness}")
    print(f"Target mean lightness: {target_mean_lightness}")
    scale = target_mean_lightness/current_mean_lightness
    l_channel = np.clip(l_channel*scale,0,255).astype(np.uint8)
    print(f"Adjusted lightness to {np.mean(l_channel)}")
    adjusted_lab_image = cv2.merge([l_channel, a_channel, b_channel])
    adjusted_image = cv2.cvtColor(adjusted_lab_image, cv2.COLOR_LAB2BGR)
    return adjusted_image

def reference_standardize(images, reference_image):
    lab_ref = cv2.cvtColor(reference_image, cv2.COLOR_BGR2LAB)
    l_ref, _, _ = cv2.split(lab_ref)

    target_lightness = np.mean(l_ref)
    standardized_images = []
    for img in images:
        standard_img = adjust_brightness(img, target_lightness)
        standardized_images.append(standard_img)
    return standardized_images

reference_image = cv2.imread('data/raw_images/724_LDLeanColour.JPG')
reference_image = white_balance(reference_image)
img_list = ['data/raw_images/1704_LdLeanColor.JPG', 'data/raw_images/1701_LdLeanColor.JPG', 'data/raw_images/2401_LdLeanColor.JPG']
ready_imgs = []
for img in img_list:
    image = cv2.imread(img)
    half = cv2.resize(image, (0,0), fx=0.15, fy=0.15)
    balance = white_balance(half)
    ready_imgs.append(balance)
standardized = reference_standardize(ready_imgs, reference_image)

for img in standardized:
    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", get_pixel_value)
    cv2.waitKey()
    cv2.destroyAllWindows()

    
    
