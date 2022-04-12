from pycocotools.coco import COCO
import json
import math
import numpy as np


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
        print(mask)
        masks.append(mask)

    return masks


if __name__ == "__main__":
    # annotation_path = r"D:\temp\coco\annotations.json"
    # instance_balance(annotation_file)
    # coco = COCO(annotation_file=annotation_path)
    # get_segmentation(coco, ['circle'], 5)
    coco = get_coco(r"D:\temp\coco\annotations.json")
    get_mask(coco, 'circle', 2)
