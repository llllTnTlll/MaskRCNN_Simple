import random

import numpy as np
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


a = np.array([[[3, 4, 5], [4, 6, 8]],
              [[3, 4, 7], [6, 8, 9]]])

b = np.array([[[2, 5], [2, 5]],
              [[2, 5], [2, 5]],
              [[2, 5], [2, 5]]])

a = np.delete(a, 1, 2)
print(a)
