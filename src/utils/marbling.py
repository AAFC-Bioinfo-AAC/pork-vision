import cv2
import numpy as np
from matplotlib import pyplot as plt


def global_threshold(image):
    '''
    A function which creates a binary image using thresholds
    Input: A Matlike image, usually through libraries like cv2.
    Output: A binary black and white image.
    '''
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(8,8))
    gray_image = clahe.apply(gray_image)
    cv2.imshow('clahe', gray_image)
    cv2.waitKey(0)
    ret, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY) #adjust first value to change sensitivity
    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    concat = np.hstack((binary_image, image))
    cv2.imshow('image', concat)
    cv2.waitKey(0)

    return binary_image

def overlay_images(image_1, image_2):
    '''
    Overlays a primary image with a secondary image.
    Input: Primary and Secondary images.
    Output: Overlay_image with the two on top of each other.
    '''
    overlay_image = cv2.addWeighted(image_1, 1.0, image_2, 0.18, 0)
    cv2.imshow('overlay image', overlay_image)
    cv2.waitKey(0)
    return overlay_image


def edge_detect(image):
    '''
    EXPERIMENTAL
    Used to detect edges with CANNY
    Input: Matlike image
    Output: Canny produced images of edges.
    '''
    edges = cv2.Canny(image, 143, 300)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.imshow('image', edges)
    cv2.waitKey(0)
    return edges_colored

def optimize_image(image):
    '''
    Adjusts b,g,r channels for an image to make it more effective for analysis
    Input: Matlike image
    Output: Image with altered color channels.
    '''
    b, g, r = cv2.split(image)
    r_modified = np.clip(255 * (r / 255) ** 1, 0, 255).astype(np.uint8)
    g_modified = np.clip( 255 * (g/ 255) ** 2.0, 0, 255).astype(np.uint8)
    b_modified = np.clip( 255 * (b/ 255) ** 1.0, 0, 255).astype(np.uint8)
    optimized_image = cv2.merge([b_modified, g_modified, r_modified])
    cv2.imshow("Optimized Image", optimized_image)
    return optimized_image

def sharpen(image):
    '''
    Applies a sharpening kernel to make the contrasting colors pop out.
    Input: Matlike image.
    Output: Sharpened image
    '''
    sharpen_kernel = np.array([[-1.0, -1.0, -1.0],
                           [-1.0,  9.0, -1.0],
                           [-1.0, -1.0, -1.0]])
    sharpened = cv2.filter2D(image, -1, sharpen_kernel)
    return sharpened
    
test = ["724_LDLeanColour.JPG", "1701_LdLeanColor.JPG", "1704_LdLeanColor.JPG", "2401_LdLeanColor.JPG"]
for data in test:
    image = cv2.imread(f"data/raw_images/{data}")
    image = cv2.resize(image, (0, 0), fx = 0.2, fy = 0.2) # Have to scale in order for the program to run on weaker hardware.
    sharpened = sharpen(image)
    optimized = optimize_image(sharpened)
    cv2.waitKey(0)
    binary = global_threshold(optimized)
    overlay = overlay_images(image, binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()