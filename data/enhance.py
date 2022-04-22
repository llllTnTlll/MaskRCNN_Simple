from pycocotools.coco import COCO
import os
import cv2 as cv
from utils import compute_bbox
from cv_helper import mask2seg, seg2mask, segs2mask, mask_iou
import random
import numpy as np
import json
from visual_ops import coco_visual


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


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

        ann_translated = cv.warpAffine(ann['foreground'], mat_translation, (image_height, image_width))
        mask_tanslated = cv.warpAffine(ann['mask'], mat_translation, (image_height, image_width))
        # cv.imshow('ann', ann_translated)
        # cv.imshow('mask', mask_tanslated)
        # cv.waitKey()

        # 修改变更的各项数据
        # coco_info: seg、bbox改变了
        coco_info['segmentation'] = mask2seg(mask_tanslated)
        coco_info['bbox'] = compute_bbox(mask_tanslated)
        # ann: ann、 mask改变了
        ann['foreground'] = ann_translated
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
        foreground = cv.copyTo(image, mask)
        anns.append(dict(
            shapes=image_shapes,
            foreground=foreground,
            mask=mask,
        ))
    return coco_infos, anns


def do_deformation(coco_info, ann, mode_list):
    deformation = Deformation()
    for mode in mode_list:
        if hasattr(deformation, mode):
            method = getattr(deformation, mode)
            coco_info, ann = method(coco_info, ann)
        else:
            ex = Exception("no such method in Deformation class!")
            raise ex
    return coco_info, ann


def enhance_apply(file_path, coco_infos, anns, mode_list):
    # 获取数据集当前的基本信息
    json_labels = json.load(open(file_path, 'r'))
    image_num = len(json_labels['images'])
    insert_id = len(json_labels["annotations"])

    targeted = False
    while not targeted:
        # 循环开始前重新载入coco
        coco = get_coco(file_path)
        # 提前生成预插入图像列表
        if image_num >= len(anns):
            target_list = random.sample(range(0, image_num), len(anns))
        else:
            target_list = random.sample(range(0, image_num), image_num)

        insert_list = []      # 待写入列表
        for i in range(len(target_list)):
            target_id = target_list.pop()
            coco_info = coco_infos.pop()
            ann = anns.pop()
            coco_info, ann = do_deformation(coco_info, ann, mode_list)

            # 获取被插入图像全局mask
            img_info = coco.loadImgs(ids=target_id)[0]
            img_shape = (img_info['height'], img_info['width'])
            ann_ids = coco.getAnnIds(imgIds=target_id)
            target_anns = coco.loadAnns(ann_ids)
            target_segs = []
            for target_ann in target_anns:
                target_segs.append(target_ann['segmentation'])
            target_mask = segs2mask(img_shape, target_segs)

            # 验证交并比
            iou = mask_iou(ann['mask'], target_mask)
            if iou == 0:
                annotation = dict(
                    id=insert_id,
                    image_id=target_id,
                    category_id=coco_info['category_id'],
                    segmentation=coco_info['segmentation'],
                    area=coco_info['area'],
                    bbox=coco_info['bbox'],
                    iscrowd=0,
                )
                insert_list.append(dict(
                    annotation=annotation,
                    foreground=ann['foreground']
                ))
                insert_id += 1
            else:
                anns.insert(0, ann)
                coco_infos.insert(0, coco_info)

        # 将待写入内容全部写入coco文件
        with open(file_path, 'r') as rf:
            params = json.load(rf)
            for insert in insert_list:
                params['annotations'].append(insert['annotation'])
                # 将拼贴应用至对应图像
                foreground = insert['foreground']
                image_name = coco.loadImgs(ids=insert['annotation']['image_id'])[0]['file_name']
                root = os.path.dirname(file_path)
                image_path = os.path.join(root, image_name)
                origin = cv.imread(image_path)
                added = cv.add(foreground, origin)
                cv.imwrite(image_path, added)
                print('added')

        with open(file_path, 'w') as wf:
            json.dump(params, wf, cls=NpEncoder)

        if len(anns) == 0:
            targeted = True


if __name__ == '__main__':
    path = r"C:\Users\zhiyuan\Desktop\temp\coco\annotations.json"
    img_folder = r"C:\Users\zhiyuan\Desktop\temp_copy\coco\JPEGImages"
    c = get_coco(path)
    _coco_infos, _anns = get_anns(c, 'rectangle', 2)
    enhance_apply(path, _coco_infos, _anns, ['translate'])
    coco_visual(path, img_folder)
