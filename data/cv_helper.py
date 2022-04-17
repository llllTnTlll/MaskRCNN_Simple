import cv2 as cv
import numpy as np


def mask2polygon(mask):
    """
    输入二值化掩膜获得polygon标注信息
    :param mask:
    :return:
    """
    contours, hierarchy = cv.findContours((mask).astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # mask_new, contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
        contour_list = contour.flatten().tolist()
        if len(contour_list) > 4:  # and cv2.contourArea(contour)>10000
            segmentation.append(contour_list)
    return segmentation


def image2binary(image):
    """大津法二值化"""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    return binary

