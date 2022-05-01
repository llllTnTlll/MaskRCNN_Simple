import json
import random

import numpy as np
from pycocotools.coco import COCO
from data.cv_helper import seg2mask
from data.cv_helper import segs2mask
import cv2 as cv

path = r"C:\Users\zhiyuan\Desktop\coco\annotations.json"
coco = COCO(annotation_file=path)
ann_ids = coco.getAnnIds(imgIds=3)
anns = coco.loadAnns(ann_ids)
for ann in anns:
    s = ann['segmentation']
    print(len(s))
    m = seg2mask((512, 512), s)
    cv.imshow('', m)
    cv.waitKey()
print(coco)
