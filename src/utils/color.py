from PIL import Image, ImageEnhance
import cv2
import numpy as np
from skimage.exposure import match_histograms,equalize_hist


original_colors = []
standard_colors = []

test_c6 = []

###################################
######TESTING FUNCTIONS############
###################################

def test_LAB(reference_image, image):
    '''
    Used to test the LAB values,
    See if the LAB of an image matches the reference LAB
    after standardization.
    '''
    lab_ref = cv2.cvtColor(reference_image, cv2.COLOR_BGR2LAB)
    l_ref, a_ref, b_ref = cv2.split(lab_ref)
    target_mean_l = np.mean(l_ref)
    target_mean_a = np.mean(a_ref)
    target_mean_b = np.mean(b_ref)
    print(f"Target mean L: {target_mean_l}")
    print(f"Target mean A: {target_mean_a}")
    print(f"Target mean B: {target_mean_b}")
    lab_current = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_cur, a_cur, b_cur = cv2.split(lab_current)
    current_mean_l = np.mean(l_cur)
    current_mean_a = np.mean(a_cur)
    current_mean_b = np.mean(b_cur)
    print(f"Current mean L: {current_mean_l}")
    print(f"Current mean A: {current_mean_a}")
    print(f"Current mean B: {current_mean_b}")


def test_var(original_colors, standard_colors):
    '''
    Prints the BGR values of the pixels extracted along with
    BGR variance between different images.
    '''
    print(f"Original values:{original_colors}")
    original_array = np.array(original_colors)
    print(f"Variance of original: {np.var(original_array, axis=0)}")
    print(f"Standardized values:{standard_colors}")
    standardized_array = np.array(standard_colors)
    print(f"Variance of Standardized: {np.var(standardized_array, axis = 0)}")

def test_extract(image, BoolStandard):
    '''
    Opens image and allows extraction of pixel values.
    '''
    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", get_pixel_value, [image, BoolStandard])
    cv2.waitKey()
    cv2.destroyAllWindows()
    print("=========================================")


def white_balance(image, option):
    '''
    Balance the white in an image
    Helps reduce lighting impact.
    Simple WB: Result is closer to the original, but increased variance.
    Learning WB: Result has less variance.
    '''
    if option == "SimpleWB":
        result = cv2.xphoto.createSimpleWB().balanceWhite(image)
    if option == "LearnWB":
        result = cv2.xphoto.createLearningBasedWB().balanceWhite(image)
    return result

def get_pixel_value(event, x, y, flag, params):
    '''
    Mainly used for testing, can be removed once a suitable method to automation is found.
    Allows a pixel to be clicked and the colors to be extracted (in BGR).
    '''
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel_value = params[0][y,x]
        print(f'Pixel value at ({x}, {y}): {pixel_value}')
        if params[1] == True:
            standard_colors.append(np.array(pixel_value).tolist())
        else:
            original_colors.append(np.array(pixel_value).tolist())

def reference_standardize(images, reference_image):
    '''
    Takes in a list of images, and a reference.
    Standardizes the list of images to the reference
    by matching histograms.
    Returns standardized image.
    '''
    standardized_images = []
    reference_image = cv2.fastNlMeansDenoisingColored(reference_image,None,3,3,7,21)
    reference_image= cv2.resize(reference_image, (0,0), fx=0.15, fy=0.15)
    reference_image = white_balance(reference_image, "LearnWB")
    for img in images:
        standard_img = match_histograms(img, reference_image, channel_axis=-1)
        standardized_images.append(standard_img)
    return standardized_images

reference_image = cv2.imread('data/raw_images/1701_LdLeanColor.JPG')

img_list = ['data/raw_images/724_LDLeanColour.JPG','data/raw_images/1704_LdLeanColor.JPG', 'data/raw_images/1701_LdLeanColor.JPG', 'data/raw_images/2401_LdLeanColor.JPG']
ready_imgs = []
for img in img_list:
    print(img)
    image = cv2.imread(img)
    half = cv2.resize(image, (0,0), fx=0.15, fy=0.15)
    test_extract(half, False)
    balance = white_balance(half, "LearnWB")
    ready_imgs.append(balance)
standardized = reference_standardize(ready_imgs, reference_image)


print("================================================================")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

for img in standardized:
    test_extract(img, True)
test_var(original_colors, standard_colors)
    

    
