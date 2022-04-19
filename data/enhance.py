from pycocotools.coco import COCO
import os
import cv2 as cv


def get_coco(annotation_path):
    coco = COCO(annotation_file=annotation_path)
    return coco


def get_path(coco: COCO, image_id):
    image_name = coco.loadImgs(image_id)[0]['file_name']
    # TODO: 根目录获取
    image_path = os.path.join(r'C:\Users\zhiyuan\Desktop\temp\coco', image_name)
    return image_path


def get_anns(coco: COCO, cat_name, quantity):
    cat_id = coco.getCatIds(catNms=cat_name)
    ann_ids = coco.getAnnIds(catIds=cat_id)

    # 反复以弥补增强数据与现有数据数量的差距
    gap = quantity - len(ann_ids)
    if gap > 0:
        for i in range(gap):
            ann_ids.append(ann_ids[i])

    else:
        ann_ids = ann_ids[:quantity]

    # 将指定特征从数据集中提取出来
    anns = coco.loadAnns(ids=ann_ids)
    mask_info = []
    for ann in anns:
        image_id = ann['image_id']
        bbox = ann['bbox']
        image = cv.imread(get_path(coco, image_id))


    return mask_info


def translate(ann, mode):
    pass
