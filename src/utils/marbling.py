import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("data/raw_images/724_LDLeanColour.JPG")
image = cv2.resize(image, (0, 0), fx = 0.2, fy = 0.2)
cv2.imshow("Gamma Corrected Image", image)
cv2.waitKey(0)

def global_threshold(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(3,1))
    gray_image = clahe.apply(gray_image)
    ret, binary_image = cv2.threshold(gray_image, 140, 255, cv2.THRESH_BINARY) #180
    concat = np.hstack((binary_image, gray_image))
    cv2.imshow('image', concat)
    cv2.waitKey(0)
    return binary_image

binary_image = global_threshold(image)

def edge_detect(image):
    edges = cv2.Canny(image, 200, 300)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.imshow('image', edges)
    cv2.waitKey(0)
    return edges_colored