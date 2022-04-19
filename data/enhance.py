from pycocotools.coco import COCO
import os
import cv2 as cv
from utils import compute_bbox
from cv_helper import mask2seg, seg2mask
import random
import numpy as np


class Deformation:
    @staticmethod
    def translate(coco_info, ann):
        [x, y, w, h] = coco_info['bbox']
        image_height = ann['shapes'][0]
        image_width = ann['shapes'][1]
        left_max = x
        right_max = image_width - x - w
        top_max = image_height - y - h
        bottom_max = y

        # 随机偏移
        dx = random.randint(-left_max, right_max)
        dy = random.randint(-bottom_max, top_max)
        mat_translation = np.float32([[1, 0, dx], [0, 1, dy]])

        ann_translated = cv.warpAffine(ann['ann'], mat_translation, (image_height, image_width))
        mask_tanslated = cv.warpAffine(ann['mask'], mat_translation, (image_height, image_width))
        # cv.imshow('ann', ann_translated)
        # cv.imshow('mask', mask_tanslated)
        # cv.waitKey()

        # 修改变更的各项数据
        # coco_info: seg、bbox改变了
        coco_info['segmentation'] = mask2seg(mask_tanslated)
        coco_info['bbox'] = compute_bbox(mask_tanslated)
        # ann: ann、 mask改变了
        ann['ann'] = ann_translated
        ann['mask'] = mask_tanslated

        return coco_info, ann


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
    coco_infos = coco.loadAnns(ids=ann_ids)
    anns = []
    for info in coco_infos:
        image_id = info['image_id']
        image_width = coco.loadImgs(image_id)[0]['width']
        image_height = coco.loadImgs(image_id)[0]['height']
        image_shapes = (image_height, image_width)
        cat_id = info['category_id']
        bbox = info['bbox']
        seg = info['segmentation']

        image = cv.imread(get_path(coco, image_id))
        mask = seg2mask(image_shapes, seg)
        ann = cv.copyTo(image, mask)
        anns.append(dict(
            shapes=image_shapes,
            ann=ann,
            mask=mask,
        ))
    return coco_infos, anns


def enhance(coco_info, ann, mode_list):
    deformation = Deformation()
    for mode in mode_list:
        if hasattr(deformation, mode):
            method = getattr(deformation, mode)
            coco_info, ann = method(coco_info, ann)
        else:
            ex = Exception("no such method in Deformation class!")
            raise ex
    return coco_info, ann


if __name__ == '__main__':
    file_path = r"C:\Users\zhiyuan\Desktop\temp\coco\annotations.json"
    c = get_coco(file_path)
    coco_infos, anns = get_anns(c, 'rectangle', 1)
    c, a = enhance(coco_infos[0], anns[0], ['translate'])

