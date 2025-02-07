import cv2
import numpy as np
from matplotlib import pyplot as plt


def global_threshold(image):
    '''
    A function which creates a binary image using thresholds, and overlays on top
    of the original image.
    Input: A Matlike image, usually through libraries like cv2.
    Output: Overlay of original image and the binary.
    '''
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(3,1))
    #gray_image = clahe.apply(gray_image)
    cv2.imshow('clahe', gray_image)
    cv2.waitKey(0)
    ret, binary_image = cv2.threshold(gray_image, 158, 255, cv2.THRESH_BINARY) #adjust first value to change sensitivity
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
    overlay_image = cv2.addWeighted(image_1, 1.0, image_2, 0.1, 0)
    cv2.imshow('overlay image', overlay_image)
    cv2.waitKey(0)
    return overlay_image


def edge_detect(image):
    edges = cv2.Canny(image, 210, 300)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.imshow('image', edges)
    cv2.waitKey(0)
    return edges_colored


image = cv2.imread("data/raw_images/724_LDLeanColour.JPG")
image = cv2.resize(image, (0, 0), fx = 0.2, fy = 0.2) # Have to scale in order for the program to run on weaker hardware.
b, g, r = cv2.split(image)
r_modified = np.clip(255 * (r / 255) ** 0.1, 0, 255).astype(np.uint8)
g_modified = np.clip( 255 * (g/ 255) ** 1.5, 0, 255).astype(np.uint8)
b_modified = np.clip( 255 * (b/ 255) ** 2.0, 0, 255).astype(np.uint8)
optimized_image = cv2.merge([b_modified, g_modified, r_modified])
cv2.imshow("Optimized Image", optimized_image)
cv2.waitKey(0)
binary_image = global_threshold(optimized_image)
overlay_image = overlay_images(image_1=image, image_2=binary_image)