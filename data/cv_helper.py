import cv2
import cv2 as cv
import numpy as np


def mask2seg(mask):
    """
    输入二值化掩膜获得polygon标注信息
    :param mask:
    :return:
    """
    contours, hierarchy = cv.findContours(mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
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


def segs2mask(img_shape, segs):
    total_mask = np.zeros(img_shape, dtype=np.uint8)
    for seg in segs:
        mask = seg2mask(img_shape, seg)
        total_mask = cv.add(total_mask, mask)
    return total_mask


def mask_iou(mask1, mask2):
    ret1, binary1 = cv.threshold(mask1, 125, 1, cv.THRESH_BINARY)
    ret2, binary2 = cv.threshold(mask2, 125, 1, cv.THRESH_BINARY)
    mask = binary1 + binary2
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

