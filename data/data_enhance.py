from pycocotools.coco import COCO
import json
import math
import numpy as np
import cv2 as cv
import os
import random
from cv_helper import *
from utils import *


def get_coco(file_path):
    coco = COCO(annotation_file=file_path)
    return coco


def get_mask(coco: COCO, cat_name, num):
    """
    从数据集中取得num个指定类的mask
    :param coco:
    :param cat_name:
    :param num:
    :return:
    """
    cat_id = coco.getCatIds(catNms=cat_name)
    ann_ids = coco.getAnnIds(catIds=cat_id)

    # TODO: 允许num溢出，重复遍历mask
    assert num <= len(ann_ids), 'there is no {} {} in the coco dataset'.format(num, cat_name)
    ann_ids = ann_ids[:num]
    anns = coco.loadAnns(ids=ann_ids)
    mask_info = []
    for ann in anns:
        mask = coco.annToMask(ann=ann)
        mask_info.append({'image_id': ann['image_id'], 'mask': mask, 'bbox': ann['bbox']})
        # print(masks)
    return mask_info


def get_path(coco: COCO, image_id):
    image_name = coco.loadImgs(image_id)[0]['file_name']
    # TODO: 根目录获取
    image_path = os.path.join(r'C:\Users\zhiyuan\Desktop\temp\coco', image_name)
    return image_path


# TODO: 改进
def instance_count(annotation_path):
    result = []
    coco = COCO(annotation_file=annotation_path)
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    for name in nms:
        cat_id = coco.getCatIds(catNms=name)
        img_ids = coco.getImgIds(catIds=cat_id)

        j = json.load(open(annotation_path, "r"))
        count = 0
        for ann in j['annotations']:
            if ann['category_id'] == cat_id[0]:
                count += 1
        # print("class {} has {} instance in {} images".format(name, count, len(img_ids)))
        result.append({'cls': name, 'count': count, 'img_ids': img_ids})
    return result


def instance_balance(annotation_path, float_range=0.2):
    counts = instance_count(annotation_path)

    # 按照实例数量进行冒泡排序
    for i in range(len(counts) - 1):
        for j in range(len(counts) - i - 1):
            if counts[j]['count'] > counts[j + 1]['count']:
                counts[j], counts[j + 1] = counts[j + 1], counts[j]

    # min_count = math.floor(counts[1]['count'] * (1-float_range))
    # max_count = math.ceil(counts[1]['count'] * (1+float_range))
    # # print(min_count, max_count)
    #
    # reduce_dic = []
    # for i in range(1, len(counts), 1):
    #     if counts[i]['count'] <
    #     reduce_dic.append({'cls': counts[i]['cls'], 'reduce': counts})


def extract_ann(coco: COCO, masks):
    """
    根据mask从对应图像中提取特征
    :param coco:
    :param masks:
    :return:
    """
    for mask in masks:
        image_id = mask['image_id']
        image = cv.imread(get_path(coco, image_id))
        ann = cv.copyTo(image, mask['mask'])
        # cv.imshow('', ann)
        # cv.waitKey()
        yield ann


def cut_paste(coco: COCO, cat_name, num, image_shape=None,
              annotation_path=r"C:\Users\zhiyuan\Desktop\temp\coco\annotations.json"):
    """
    coco数据集数据增强
    根据mask进行裁剪拼贴
    自动标注
    :param annotation_path:
    :param image_shape:
    :param coco:
    :param cat_name:
    :param num:
    :return:
    """
    if image_shape is None:
        image_shape = [512, 512, 3]
    image_wide = image_shape[0]
    image_height = image_shape[1]

    mask_info = get_mask(coco, cat_name, num)
    anns = extract_ann(coco, mask_info)
    for i in range(num):
        ann = next(anns)
        info = mask_info[i]
        # 根据bbox推算平移变换范围
        [x, y, w, h] = info['bbox']
        left_max = x
        right_max = image_wide - x - w
        top_max = image_height - y - h
        bottom_max = y

        # 随机偏移
        dx = random.randint(-left_max, right_max)
        dy = random.randint(-bottom_max, top_max)
        mat_translation = np.float32([[1, 0, dx], [0, 1, dy]])

        image_moved = cv.warpAffine(ann, mat_translation, (image_wide, image_height))
        mask = image2binary(image_moved)
        bbox = compute_bbox(mask)
        seg = mask2polygon(mask)

        # cv.imshow('image_moved', image_moved)
        # cv.imshow('mask', mask)
        # print(bbox)
        # print(seg)
        # cv.waitKey()

        ann_idx = coco.getAnnIds(imgIds=i)
        objects = coco.loadAnns(ann_idx)

        annotations = []
        start_id = len(json.load(open(annotation_path, "r"))['annotations'])
        overlap = False
        for obj in objects:
            obj_bbox = obj['bbox']
            iou = compute_iou(wh2xy(bbox), wh2xy(obj_bbox))
            # src = cv.imread(get_path(coco, i))
            print(iou)
            if iou > 0:
                overlap = True
                break

        if not overlap:
            annotations.append(
                dict(
                    id=start_id,
                    image_id=image_id,
                    category_id=cls_id,
                    segmentation=segmentations[instance],
                    area=area,
                    bbox=bbox,
                    iscrowd=0,
                )
            )


if __name__ == "__main__":
    coco = get_coco(r"C:\Users\zhiyuan\Desktop\temp\coco\annotations.json")
    cut_paste(coco, 'rectangle', 8)
    # ids = coco.getAnnIds(imgIds=1)
    # anns = coco.loadAnns(ids)
    # for ann in anns:
    #     seg = np.reshape(ann['segmentation'], (-1, 2))
    #     poly = np.array(seg, np.int32)
    #
    #     img = cv.imread(get_path(coco, image_id=ann['image_id']))
    #     mask = cv.fillPoly(np.zeros(img.shape, img.dtype), [poly], (255, 255, 255))
    #     mask = image2binary(image=mask)
    #     bbox = compute_bbox(mask)
    #     true_box = ann['bbox']
    #     print(bbox, true_box)
