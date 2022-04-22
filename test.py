import random
from pycocotools.coco import COCO
from data.cv_helper import segs2mask
import cv2 as cv

# path = r"C:\Users\zhiyuan\Desktop\temp\coco\annotations.json"
# coco = COCO(annotation_file=path)
# image_id = 1
# ann_ids = coco.getAnnIds(imgIds=image_id)
# anns = coco.loadAnns(ann_ids)
# segs = []
# for ann in anns:
#     segs.append(ann['segmentation'])
# mask = segs2mask((512, 512), segs)
# cv.imshow('', mask)
# cv.waitKey()
img = cv.imread(r"C:\Users\zhiyuan\Desktop\temp\coco\JPEGImages\2.jpg", cv.COLOR_BGR2RGB)
cv.imshow('', img)
cv.waitKey()