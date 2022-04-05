import skimage.io as io
import logging
import random

import numpy as np
import skimage.color
import skimage.io
import skimage.transform

import utils
from anchor_ops import compute_backbone_shapes, generate_pyramid_anchors
from layers import build_rpn_targets

from PIL import Image
import yaml
import os

import collections
import datetime
import glob
import json

import os.path as osp
import uuid

import imgviz
import labelme
import pycocotools.mask
from pycocotools.coco import COCO
import cv2


# yaml
class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, class_map=None):
        # 从0开始的一维数组对应图像索引
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.

        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        return ""

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's available online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].

        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        logging.warning("You are using the default load_mask(), maybe you need to define your own one.")
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids


class ShapesDataset(Dataset):
    # 得到该图中有多少个实例（物体）
    @staticmethod
    def get_obj_index(image):
        n = np.max(image)
        return n

    # 解析labelme中得到的yaml文件，从而得到mask每一层对应的实例标签
    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read(), Loader=yaml.FullLoader)
            labels = temp['label_names']
            del labels[0]
        return labels

    # 重新写draw_mask
    def draw_mask(self, num_obj, mask, image, image_id):
        info = self.image_info[image_id]
        for index in range(num_obj):
            for i in range(np.shape(mask)[1]):
                for j in range(np.shape(mask)[0]):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask

    # 并在self.image_info信息中添加了path、mask_path 、yaml_path
    def load_shapes(self, count, img_floder, mask_floder, imglist, yaml_floder):
        # TODO: 修改此处
        self.add_class("shapes", 1, "circle")
        self.add_class("shapes", 2, "square")
        self.add_class("shapes", 3, "triangle")
        for i in range(count):
            img = imglist[i]
            if img.endswith(".jpg"):
                img_name = img.split(".")[0]
                img_path = os.path.join(img_floder, img)
                mask_path = os.path.join(mask_floder, img_name) + ".png"
                yaml_path = os.path.join(yaml_floder, img_name) + ".yaml"
                self.add_image("shapes", image_id=i, path=img_path, mask_path=mask_path, yaml_path=yaml_path)

    # 重写load_mask
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([np.shape(img)[0], np.shape(img)[1], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        labels = []
        labels = self.from_yaml_get_class(image_id)
        labels_form = []
        for i in range(len(labels)):
            # TODO:修改此处
            if labels[i].find("circle") != -1:
                labels_form.append("circle")
            elif labels[i].find("square") != -1:
                labels_form.append("square")
            elif labels[i].find("triangle") != -1:
                labels_form.append("triangle")
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)


def load_image_gt(dataset, config, image_id, augment=False, augmentation=None,
                  use_mini_mask=False):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

    augment: (deprecated. Use augmentation instead). If true, apply random
        image augmentation. Currently, only horizontal flipping is offered.
    augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
        For example, passing imgaug.augmenters.Fliplr(0.5) flips images
        right/left 50% of the time.
    use_mini_mask: If False, returns full-size masks that are the same height
        and width as the original image. These can be big, for example
        1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
        224x224 and are generated by extracting the bounding box of the
        object and resizing it to MINI_MASK_SHAPE.

    Returns:
    image: [height, width, 3]
    shape: the original shape of the image before resizing and cropping.
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (y1, x1, y2, x2)]
    mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """
    # Load image and mask
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    original_shape = image.shape
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    mask = utils.resize_mask(mask, scale, padding, crop)

    # Random horizontal flips.
    # TODO: will be removed in a future update in favor of augmentation
    if augment:
        logging.warning("'augment' is deprecated. Use 'augmentation' instead.")
        if random.randint(0, 1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)

    # Augmentation
    # This requires the imgaug lib (https://github.com/aleju/imgaug)
    if augmentation:
        import imgaug

        # Augmenters that are safe to apply to masks
        # Some, such as Affine, have settings that make them unsafe, so always
        # test your augmentation on masks
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        # Store shapes before augmentation to compare
        image_shape = image.shape
        mask_shape = mask.shape
        # Make augmenters deterministic to apply similarly to images and masks
        det = augmentation.to_deterministic()
        image = det.augment_image(image)
        # Change mask to np.uint8 because imgaug doesn't support np.bool
        mask = det.augment_image(mask.astype(np.uint8),
                                 hooks=imgaug.HooksImages(activator=hook))
        # Verify that shapes didn't change
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
        # Change mask back to bool
        mask = mask.astype(np.bool)

    # Note that some boxes might be all zeros if the corresponding mask got cropped out.
    # and here is to filter them out
    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask = mask[:, :, _idx]
    class_ids = class_ids[_idx]
    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = utils.extract_bboxes(mask)

    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1

    # Resize masks to smaller size to reduce memory usage
    if use_mini_mask:
        mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

    # Image meta data
    image_meta = utils.compose_image_meta(image_id, original_shape, image.shape,
                                          window, scale, active_class_ids)

    return image, image_meta, class_ids, bbox, mask


def data_generator(dataset, config, shuffle=True, augment=False, augmentation=None,
                   batch_size=1, detection_targets=False,
                   no_augmentation_sources=None):
    """
    网络输入清单
    - images: [batch, H, W, C]
    - image_meta: [batch, (meta data)] 图像详细信息。
    - rpn_match: [batch, N] 代表建议框的匹配情况 (1=正样本, -1=负样本, 0=中性)
    - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] 建议框网络应该有的预测结果.
    - gt_class_ids: [batch, MAX_GT_INSTANCES] 种类ID
    - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
    - gt_masks: [batch, height, width, MAX_GT_INSTANCES].

    网络输出清单:
        在常规训练中通常是空的。
    """
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    no_augmentation_sources = no_augmentation_sources or []

    # [anchor_count, (y1, x1, y2, x2)]
    # 计算获得先验框
    backbone_shapes = compute_backbone_shapes(config.IMAGE_SHAPE, config.FRATURE_STRIDES)
    anchors = generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                       config.RPN_ANCHOR_RATIOS,
                                       backbone_shapes,
                                       config.FRATURE_STRIDES,
                                       config.RPN_ANCHOR_STRIDE)

    while True:

        image_index = (image_index + 1) % len(image_ids)
        if shuffle and image_index == 0:
            np.random.shuffle(image_ids)

        # 获得id
        image_id = image_ids[image_index]

        # 获得图片，真实框，语义分割结果等
        if dataset.image_info[image_id]['source'] in no_augmentation_sources:
            image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                load_image_gt(dataset, config, image_id, augment=augment,
                              augmentation=None,
                              use_mini_mask=config.USE_MINI_MASK)
        else:
            image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                load_image_gt(dataset, config, image_id, augment=augment,
                              augmentation=augmentation,
                              use_mini_mask=config.USE_MINI_MASK)

        if not np.any(gt_class_ids > 0):
            continue

        # RPN Targets
        # anchors, gt_boxes, image_shape, batch_size, rpn_train_anchors_per_image, rpn_bbox_std_dev
        rpn_match, rpn_bbox = build_rpn_targets(anchors=anchors,
                                                gt_boxes=gt_boxes,
                                                image_shape=config.IMAGE_SHAPE[0:2],
                                                batch_size=config.BATCH_SIZE,
                                                rpn_train_anchors_per_image=config.RPN_TRAIN_ANCHORS_PER_IMAGE,
                                                rpn_bbox_std_dev=config.RPN_BBOX_STD_DEV)

        # 如果某张图片里面物体的数量大于最大值的话，则进行筛选，防止过大
        if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
            ids = np.random.choice(
                np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
            gt_class_ids = gt_class_ids[ids]
            gt_boxes = gt_boxes[ids]
            gt_masks = gt_masks[:, :, ids]

        # 初始化用于训练的内容
        if b == 0:
            batch_image_meta = np.zeros(
                (batch_size,) + image_meta.shape, dtype=image_meta.dtype)
            batch_rpn_match = np.zeros(
                [batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
            batch_rpn_bbox = np.zeros(
                [batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
            batch_images = np.zeros(
                (batch_size,) + image.shape, dtype=np.float32)
            batch_gt_class_ids = np.zeros(
                (batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
            batch_gt_boxes = np.zeros(
                (batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)
            batch_gt_masks = np.zeros(
                (batch_size, gt_masks.shape[0], gt_masks.shape[1],
                 config.MAX_GT_INSTANCES), dtype=gt_masks.dtype)

        # 将当前信息加载进batch
        batch_image_meta[b] = image_meta
        batch_rpn_match[b] = rpn_match[:, np.newaxis]
        batch_rpn_bbox[b] = rpn_bbox
        batch_images[b] = utils.mold_image(image.astype(np.float32), config)
        batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
        batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
        batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks

        b += 1

        # 判断是否已经将batch_size全部载入
        if b >= batch_size:
            inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                      batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
            outputs = []

            yield inputs, outputs
            # 开始一个新的batch_size
            b = 0


# coco
class CoCoDataGenrator:
    def __init__(self, config):

        # coco_annotation_file,
        # img_shape = (640, 640, 3),
        # batch_size = 1,
        # max_instances = 100,
        # include_crowd = False,
        # include_mask = True,
        # include_keypoint = False,
        # image_mean = np.array([[[102.9801, 115.9465, 122.7717]]]),
        # use_mini_mask = True,
        # mini_mask_shape = (56, 56),
        # data_size = -1

        self.coco_annotation_file = config.COCO_ANNOTATION_FILE
        self.img_shape = config.IMAGE_SHAPE
        self.batch_size = config.BATCH_SIZE
        self.max_instances = config.DETECTION_MAX_INSTANCES
        self.include_crowd = config.COCO_INCLUDE_CROWD
        self.include_mask = config.C0C0_INCLUDE_MASK
        self.include_keypoint = config.COCO_INCLUDE_KEYPOINT
        self.img_mean = config.IMAGE_MEAN
        self.use_mini_mask = config.USE_MINI_MASK
        self.mini_mask_shape = config.MINI_MASK_SHAPE
        self.data_size = config.COCO_DATA_SIZE

        self.current_batch_index = 0
        self.total_batch_size = 0
        self.img_ids = []
        self.coco = COCO(annotation_file=self.coco_annotation_file)
        self.load_data()

    def load_data(self):
        # 初步过滤数据是否包含crowd
        target_img_ids = []
        for k in self.coco.imgToAnns:
            annos = self.coco.imgToAnns[k]
            if annos:
                annos = list(filter(lambda x: x['iscrowd'] == self.include_crowd, annos))
                if annos:
                    target_img_ids.append(k)
        # 使用数据限制
        if self.data_size >= 0:
           target_img_ids = target_img_ids[:self.data_size]

        # TODO:与config同步
        self.total_batch_size = len(target_img_ids) // self.batch_size
        self.img_ids = target_img_ids

    def next_batch(self):
        # 判断是否进行至本轮结束
        if self.current_batch_index >= self.total_batch_size:
            self.current_batch_index = 0
            self._on_epoch_end()

        batch_img_ids = self.img_ids[self.current_batch_index * self.batch_size:
                                     (self.current_batch_index + 1) * self.batch_size]
        batch_imgs = []
        batch_bboxes = []
        batch_labels = []
        batch_masks = []
        batch_keypoints = []
        valid_nums = []

        for img_id in batch_img_ids:
            # {"img":, "bboxes":, "labels":, "masks":, "key_points":}
            data = self._data_generation(image_id=img_id)
            if len(np.shape(data['img'])) > 0:
                batch_imgs.append(data['img'])
                batch_labels.append(data['labels'])
                batch_bboxes.append(data['bboxes'])
                valid_nums.append(data['valid_nums'])
                # if len(data['labels']) > self.max_instances:
                #     batch_bboxes.append(data['bboxes'][:self.max_instances, :])
                #     batch_labels.append(data['labels'][:self.max_instances])
                #     valid_nums.append(self.max_instances)
                # else:
                #     pad_num = self.max_instances - len(data['labels'])
                #     batch_bboxes.append(np.pad(data['bboxes'], [(0, pad_num), (0, 0)]))
                #     batch_labels.append(np.pad(data['labels'], [(0, pad_num)]))
                #     valid_nums.append(len(data['labels']))

                if self.include_mask:
                    batch_masks.append(data['masks'])

                if self.include_keypoint:
                    batch_keypoints.append(data['keypoints'])

        self.current_batch_index += 1
        if len(batch_imgs) < self.batch_size:
            return self.next_batch()

        output = {
            'imgs': np.array(batch_imgs, dtype=np.float32),
            'bboxes': np.array(batch_bboxes, dtype=np.float32),
            'labels': np.array(batch_labels, dtype=np.int8),
            'masks': np.array(batch_masks, dtype=np.int8),
            'keypoints': np.array(batch_keypoints, dtype=np.float32),
            'valid_nums': np.array(valid_nums, dtype=np.int8)
        }

        if self.include_mask:
            return output['imgs'], output['masks'], output['bboxes'], output['labels']
        return output

    def _on_epoch_end(self):
        np.random.shuffle(self.img_ids)

    def _resize_im(self, origin_im, bboxes):
        """ 对图片/mask/box resize

        :param origin_im
        :param bboxes
        :return im_blob: [h, w, 3]
                gt_boxes: [N, [ymin, xmin, ymax, xmax]]
        """
        im_shape = np.shape(origin_im)
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(self.img_shape[0]) / float(im_size_max)

        # resize原始图片
        im_resize = cv2.resize(origin_im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        im_resize_shape = np.shape(im_resize)
        im_blob = np.zeros(self.img_shape, dtype=np.float32)
        im_blob[0:im_resize_shape[0], 0:im_resize_shape[1], :] = im_resize

        # resize对应边框
        bboxes_resize = np.array(bboxes * im_scale, dtype=np.float32)

        return im_blob, bboxes_resize

    def _resise_mini_mask(self, masks, boxes):
        """  mask处理成最小mask """
        mini_masks = []
        h, w, c = np.shape(masks)
        for i in range(c):
            ymin, xmin, ymax, xmax = boxes[i]
            mask = masks[int(ymin):int(ymax), int(xmin):int(xmax), i]
            mini_m = cv2.resize(mask, self.mini_mask_shape, interpolation=cv2.INTER_LINEAR)
            mini_m = np.array(mini_m >= 0.5, dtype=np.int8)
            mini_m = np.expand_dims(mini_m, axis=-1)
            mini_masks.append(mini_m)
        mini_masks = np.concatenate(mini_masks, axis=-1)
        return mini_masks

    def _resize_mask(self, origin_masks):
        """ resize mask数据
        :param origin_mask:
        :return: mask_resize: [h, w, instance]
                 gt_boxes: [N, [ymin, xmin, ymax, xmax]]
        """
        mask_shape = np.shape(origin_masks)
        mask_size_max = np.max(mask_shape[0:3])
        im_scale = float(self.img_shape[0]) / float(mask_size_max)

        # resize mask/box
        gt_boxes = []
        masks_resize = []
        for m in origin_masks:
            m = np.array(m, dtype=np.float32)
            m_resize = cv2.resize(m, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
            m_resize = np.array(m_resize >= 0.5, dtype=np.int8)

            # 计算bdbox
            h, w = np.shape(m_resize)
            rows, cols = np.where(m_resize)
            # [xmin, ymin, xmax, ymax]
            xmin = np.min(cols) if np.min(cols) >= 0 else 0
            ymin = np.min(rows) if np.min(rows) >= 0 else 0
            xmax = np.max(cols) if np.max(cols) <= w else w
            ymax = np.max(rows) if np.max(rows) <= h else h
            bdbox = [ymin, xmin, ymax, xmax]
            gt_boxes.append(bdbox)

            mask_blob = np.zeros((self.img_shape[0], self.img_shape[1], 1), dtype=np.float32)
            mask_blob[0:h, 0:w, 0] = m_resize
            masks_resize.append(mask_blob)

        # [instance_num, [xmin, ymin, xmax, ymax]]
        gt_boxes = np.array(gt_boxes, dtype=np.int16)
        # [h, w, instance_num]
        masks_resize = np.concatenate(masks_resize, axis=-1)

        return masks_resize, gt_boxes

    def _data_generation(self, image_id):
        """ 拉取coco标记数据, 目标边框, 类别, mask
        :param image_id:
        :return:
        """

        anno_ids = self.coco.getAnnIds(imgIds=image_id, iscrowd=self.include_crowd)
        bboxes = []
        labels = []
        masks = []
        keypoints = []

        for i in anno_ids:
            # 边框, 处理成左上右下坐标
            ann = self.coco.anns[i]
            bbox = ann['bbox']
            xmin, ymin, w, h = bbox
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmin + w)
            ymax = int(ymin + h)
            bboxes.append([ymin, xmin, ymax, xmax])
            # 类别ID
            label = ann['category_id']
            labels.append(label)
            # 实例分割
            if self.include_mask:
                # [instances, h, w]
                mask = self.coco.annToMask(ann)
                # cv2.imshow("mask", np.array(mask,dtype=np.uint8)*255)
                # cv2.imshow("img", img)
                # cv2.waitKey(0)
                masks.append(mask)
            if self.include_keypoint and ann.get('keypoints'):
                keypoint = ann['keypoints']
                # 处理成[x,y,v] 其中v=0表示没有此点,v=1表示被挡不可见,v=2表示可见
                keypoint = np.reshape(keypoint, [-1, 3])
                keypoints.append(keypoint)

        # 输出包含5个东西, 不需要则为空
        outputs = {
            "img": [],
            "labels": [],
            "bboxes": [],
            "masks": [],
            "keypoints": [],
            "valid_nums": 0
        }

        valid_nums = 0
        if len(labels) > self.max_instances:
            bboxes = bboxes[:self.max_instances, :]
            labels = labels[:self.max_instances]
            valid_nums = self.max_instances
            # batch_bboxes.append(data['bboxes'][:self.max_instances, :])
            # batch_labels.append(data['labels'][:self.max_instances])
            # valid_nums.append(self.max_instances)
        else:
            pad_num = self.max_instances - len(labels)
            bboxes = np.pad(bboxes, [(0, pad_num), (0, 0)])
            labels = np.pad(labels, [(0, pad_num)])
            valid_nums = self.max_instances - pad_num
            # batch_bboxes.append(np.pad(data['bboxes'], [(0, pad_num), (0, 0)]))
            # batch_labels.append(np.pad(data['labels'], [(0, pad_num)]))
            # valid_nums.append(len(data['labels']))

        # 处理最终数据 mask
        if self.include_mask:
            # [h, w, instances]
            masks, mask_boxes = self._resize_mask(origin_masks=masks)
            # mini mask
            if self.use_mini_mask:
                masks = self._resise_mini_mask(masks, mask_boxes)
            if np.shape(masks)[2] > self.max_instances:
                masks = masks[:self.max_instances, :, :]
            else:
                pad_num = self.max_instances - np.shape(masks)[2]
                masks = np.pad(masks, [(0, 0), (0, 0), (0, pad_num)])

            outputs['masks'] = masks
            # outputs['bboxes'] = bboxes

        # 处理最终数据 keypoint
        if self.include_keypoint:
            keypoints = np.array(keypoints, dtype=np.int8)
            outputs['keypoints'] = keypoints

        img_coco_url_file = str(self.coco.imgs[image_id].get('coco_url',""))
        img_url_file = str(self.coco.imgs[image_id].get('url',""))
        img_local_file = str(self.coco.imgs[image_id].get('file_name',""))
        img_local_file = os.path.join(os.path.dirname(self.coco_annotation_file), img_local_file)
        img = []

        if os.path.isfile(img_local_file):
            img = io.imread(img_local_file)
        elif img_coco_url_file.startswith("http"):
            img = io.imread(self.coco.imgs[image_id]['coco_url'])
        elif img_url_file.startswith("http"):
            img = io.imread(self.coco.imgs[image_id]['coco_url'])
        else:
            return outputs
        if len(np.shape(img)) < 2:
            return outputs
        elif len(np.shape(img)) == 2:
            img = np.expand_dims(img, axis=-1)
            img = np.pad(img, [(0, 0), (0, 0), (0, 2)])
        else:
            img = img[:, :, ::-1]

        labels = np.array(labels, dtype=np.int8)
        bboxes = np.array(bboxes, dtype=np.float32)
        img_resize, bboxes_resize = self._resize_im(origin_im=img, bboxes=bboxes)

        outputs['img'] = img_resize - self.img_mean
        outputs['labels'] = labels
        outputs['bboxes'] = bboxes_resize
        outputs['valid_nums'] = valid_nums

        return outputs


def labelme2coco(input_dir, output_dir, labels, noviz=False):
    """
    将labelme生成的数据转换为标准coco格式
    :param input_dir: json文件目录
    :param output_dir: 输出文件目录
    :param labels: 类别txt文档
    :param noviz: 是否生成可视化图像
    :return:
    """

    now = datetime.datetime.now()

    # 建立coco数据集基本结构
    # 只关注images,annotations,categories
    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[dict(url=None, id=0, name=None,)],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type="instances",
        # 目标标注
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        # 与id对应的种类名
        categories=[
            # supercategory, id, name
        ],
    )

    # 通过读取类别txt获取类别id和name
    class_name_to_id = {}
    for i, line in enumerate(open(labels).readlines()):
        class_id = i
        class_name = line.strip()
        # if class_id == -1:
        #     assert class_name == "_background_"
        #     continue
        class_name_to_id[class_name] = class_id
        data["categories"].append(
            dict(supercategory=None, id=class_id, name=class_name,)
        )

    out_ann_file = osp.join(output_dir, "annotations.json")   # 输出目录
    label_files = glob.glob(osp.join(input_dir, "*.json"))    # 输入json目录
    for image_id, filename in enumerate(label_files):
        print("Generating dataset from:", filename)

        label_file = labelme.LabelFile(filename=filename)

        base = osp.splitext(osp.basename(filename))[0]
        out_img_file = osp.join(output_dir, "JPEGImages", base + ".jpg")

        img = labelme.utils.img_data_to_arr(label_file.imageData)
        imgviz.io.imsave(out_img_file, img)
        data["images"].append(
            dict(
                license=0,
                url=None,
                file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
                # file_name=out_img_file,
                height=img.shape[0],
                width=img.shape[1],
                date_captured=None,
                id=image_id,
            )
        )

        masks = {}  # for area
        segmentations = collections.defaultdict(list)  # for segmentation
        for shape in label_file.shapes:
            points = shape["points"]
            label = shape["label"]
            group_id = shape.get("group_id")
            shape_type = shape.get("shape_type", "polygon")
            if shape_type == "polygon":
                mask = labelme.utils.shape_to_mask(
                    img.shape[:2], points, shape_type
                )
                # cv2.imshow("",np.array(mask, dtype=np.uint8)*255)
                # cv2.waitKey(0)

                if group_id is None:
                    group_id = uuid.uuid1()

                instance = (label, group_id)
                # print(instance)

                if instance in masks:
                    masks[instance] = masks[instance] | mask
                else:
                    masks[instance] = mask
                # print(masks[instance].shape)

                if shape_type == "rectangle":
                    (x1, y1), (x2, y2) = points
                    x1, x2 = sorted([x1, x2])
                    y1, y2 = sorted([y1, y2])
                    points = [x1, y1, x2, y1, x2, y2, x1, y2]
                if shape_type == "circle":
                    (x1, y1), (x2, y2) = points
                    r = np.linalg.norm([x2 - x1, y2 - y1])
                    # r(1-cos(a/2))<x, a=2*pi/N => N>pi/arccos(1-x/r)
                    # x: tolerance of the gap between the arc and the line segment
                    n_points_circle = max(int(np.pi / np.arccos(1 - 1 / r)), 12)
                    i = np.arange(n_points_circle)
                    x = x1 + r * np.sin(2 * np.pi / n_points_circle * i)
                    y = y1 + r * np.cos(2 * np.pi / n_points_circle * i)
                    points = np.stack((x, y), axis=1).flatten().tolist()
                else:
                    points = np.asarray(points).flatten().tolist()

                segmentations[instance].append(points)
                print(segmentations[instance])
        segmentations = dict(segmentations)

        for instance, mask in masks.items():
            cls_name, group_id = instance
            if cls_name not in class_name_to_id:
                continue
            cls_id = class_name_to_id[cls_name]

            mask = np.asfortranarray(mask.astype(np.uint8))
            mask = pycocotools.mask.encode(mask)
            area = float(pycocotools.mask.area(mask))
            bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

            data["annotations"].append(
                dict(
                    id=len(data["annotations"]),
                    image_id=image_id,
                    category_id=cls_id,
                    segmentation=segmentations[instance],
                    area=area,
                    bbox=bbox,
                    iscrowd=0,
                )
            )

        if not noviz:
            viz = img
            if masks:
                labels, captions, masks = zip(
                    *[
                        (class_name_to_id[cnm], cnm, msk)
                        for (cnm, gid), msk in masks.items()
                        if cnm in class_name_to_id
                    ]
                )
                viz = imgviz.instances2rgb(
                    image=img,
                    labels=labels,
                    masks=masks,
                    captions=captions,
                    font_size=15,
                    line_width=2,
                )
            out_viz_file = osp.join(
                output_dir, "Visualization", base + ".jpg"
            )
            imgviz.io.imsave(out_viz_file, viz)

    with open(out_ann_file, "w") as f:
        json.dump(data, f)