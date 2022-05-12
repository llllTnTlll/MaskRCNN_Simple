import json
import random

import numpy as np
from pycocotools.coco import COCO
from data.cv_helper import seg2mask
from data.cv_helper import segs2mask
import cv2 as cv
import matplotlib.pyplot as plt

path = r'C:\Users\zhiyuan\Desktop\pic1.png'
img = cv.imread(path)
plt.imshow(img)
plt.show()