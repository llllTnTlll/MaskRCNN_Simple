import random
from pycocotools.coco import COCO
from data.cv_helper import segs2mask
import cv2 as cv

path = r"C:\Users\zhiyuan\Desktop\temp\coco\annotations.json"
coco = COCO(annotation_file=path)
for i in range(11):
    # 获取被插入图像全局mask
    img_info = coco.loadImgs(i)[0]
    img_shape = (img_info['height'], img_info['width'])
    ann_ids = coco.getAnnIds(imgIds=i)
    target_anns = coco.loadAnns(ann_ids)
    target_segs = []
    for target_ann in target_anns:
        target_segs.append(target_ann['segmentation'])
    target_mask = segs2mask(img_shape, target_segs)
    cv.imshow('', target_mask)
    cv.waitKey()