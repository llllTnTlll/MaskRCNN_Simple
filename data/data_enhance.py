from pycocotools.coco import COCO
import json
import math
import numpy as np
import cv2 as cv
import os
import random


def get_coco(file_path):
    coco = COCO(annotation_file=file_path)
    return coco


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

    assert num <= len(ann_ids), 'there is no {} {} in the coco dataset'.format(num, cat_name)
    ann_ids = ann_ids[:num]
    anns = coco.loadAnns(ids=ann_ids)
    masks = []
    for ann in anns:
        mask = coco.annToMask(ann=ann)
        masks.append({'mask': mask, 'image_id': ann['image_id'], 'bbox': ann['bbox']})
        # print(masks)
    return masks


def extract_ann(coco:COCO, masks):
    for mask in masks:
        image_id = mask['image_id']
        image_name = coco.loadImgs(image_id)[0]['file_name']
        # TODO: 根目录获取
        image_path = os.path.join(r'C:\Users\zhiyuan\Desktop\temp\coco', image_name)
        image = cv.imread(image_path)
        ann = cv.copyTo(image, mask['mask'])
        # cv.imshow('', ann)
        # cv.waitKey()
        yield ann


def cut_paste(coco:COCO, cat_name, num, image_shape=None):
    """
    coco数据集数据增强
    根据mask进行裁剪拼贴
    自动标注
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

    masks = get_mask(coco, cat_name, num)
    anns = extract_ann(coco, masks)
    for i in range(num):
        ann = next(anns)
        mask = masks[i]
        # 根据bbox推算平移变换范围
        [x, y, w, h] = mask['bbox']
        left_max = x
        right_max = image_wide - x - w
        top_max = image_height - y - h
        bottom_max = y

        # 应用随机偏移
        dx = random.randint(-left_max, right_max)
        dy = random.randint(-bottom_max, top_max)
        mat_translation = np.float32([[1, 0, dx], [0, 1, dy]])
        image_move_0 = cv.warpAffine(ann, mat_translation, (image_wide, image_height))

        cv.imshow('', image_move_0)
        cv.waitKey()


if __name__ == "__main__":
    coco = get_coco(r"C:\Users\zhiyuan\Desktop\temp\coco\annotations.json")
    cut_paste(coco, 'rectangle', 6)
    # masks = get_mask(coco, 'rectangles', 3)
    # extract_ann(coco, masks)
