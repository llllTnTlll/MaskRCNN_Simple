import cv2 as cv
import numpy as np


def mask2seg(mask):
    """
    输入二值化掩膜获得polygon标注信息
    :param mask:
    :return:
    """
    contours, hierarchy = cv.findContours(mask.astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # mask_new, contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
        contour_list = contour.flatten().tolist()
        if len(contour_list) > 4:  # and cv2.contourArea(contour)>10000
            segmentation.append(contour_list)
    return segmentation


def seg2mask(img_shape, seg):
    polygons = np.reshape(seg, (-1, 2))
    mask = np.zeros(img_shape, dtype=np.uint8)
    polygons = np.asarray([polygons], np.int32)  # 这里必须是int32，其他类型使用fillPoly会报错
    cv.fillConvexPoly(mask, polygons, 255)
    return mask


def seg_iou(seg1, seg2, image_shape):
    polygon1 = np.reshape(seg1, (-1, 2))
    polygon2 = np.reshape(seg2, (-1, 2))
    data1 = np.array(polygon1, np.int32)
    data2 = np.array(polygon2, np.int32)

    mask1 = np.zeros(image_shape, np.uint8)
    mask2 = np.zeros(image_shape, np.uint8)

    cv.fillPoly(mask1, [data1], 1)
    cv.fillPoly(mask2, [data2], 1)
    mask = mask1 + mask2
    inter = (mask == 2).sum()

    if inter > 0:
        return inter
    else:
        return 0


def image2binary(image):
    """大津法二值化"""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    return binary
