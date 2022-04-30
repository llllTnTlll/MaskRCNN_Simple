import tensorflow as tf
import numpy as np
import os


def tensor_feature(value):
    """ tensor序列化成feature
    :param value:
    :return:
    """
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
    )


def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
    )


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def generate_coco_segment_tfrecord(config, is_training=True, tfrec_path='../coco_tfrec/', coco_annotation_file='./voc2012_46_samples'):
    from data.generate_coco_data import CoCoDataGenrator
    from model.layers import build_rpn_targets
    from model.anchors_ops import get_anchors

    if not os.path.isdir(tfrec_path):
        os.mkdir(tfrec_path)

    coco_seg = CoCoDataGenrator(
        coco_annotation_file=coco_annotation_file,
        img_shape=config.IMAGE_SHAPE,
        batch_size=config.BATCH_SIZE,
        max_instances=config.DETECTION_MAX_INSTANCES,
        image_mean=config.PIXEL_MEAN,
        use_mini_mask=config.USE_MINI_MASK,
        mini_mask_shape=config.MINI_MASK_SHAPE,
    )

    if is_training:
        tfrec_file = os.path.join(tfrec_path, "coco_train_seg.tfrec")
    else:
        tfrec_file = os.path.join(tfrec_path, "coco_test_seg.tfrec")
    tfrec_writer = tf.io.TFRecordWriter(tfrec_file)

    anchors = get_anchors(image_shape=config.IMAGE_SHAPE,
                          scales=config.ANCHOR_SCALES,
                          ratios=config.ANCHOR_RATIOS,
                          feature_strides=config.FEATURE_STRIDE,
                          anchor_stride=config.ANCHOR_STRIDE)

    for i in range(coco_seg.total_batch_size):
        print("current {} total {}".format(i, coco_seg.total_batch_size))
        # for batch in range(vsg_train.total_batch_size):
        imgs, masks, gt_boxes, labels = coco_seg.next_batch()
        # indices = voc_seg.file_indices[i * 1: (i + 1) * 1]
        # cur_img_files = [voc_seg.img_files[k] for k in indices]
        # cur_cls_files = [voc_seg.cls_mask_files[k] for k in indices]
        # cur_obj_files = [voc_seg.obj_mask_files[k] for k in indices]
        # imgs, masks, gt_boxes, labels = voc_seg._data_generation(img_files=cur_img_files,
        #                                                          cls_files=cur_cls_files,
        #                                                          obj_files=cur_obj_files)
        rpn_target_match, rpn_target_box = build_rpn_targets(anchors=anchors,
                                                             gt_boxes=gt_boxes,
                                                             image_shape=config.IMAGE_SHAPE[:2],
                                                             batch_size=config.BATCH_SIZE,
                                                             rpn_train_anchors_per_image=config.RPN_TRAIN_ANCHORS_PER_IMAGE,
                                                             rpn_bbox_std_dev=config.BBOX_STD_DEV)
        feature = {
            "image": tensor_feature(imgs),
            "masks": tensor_feature(masks),
            "gt_boxes": tensor_feature(gt_boxes),
            "labels": tensor_feature(labels),
            "rpn_target_match": tensor_feature(rpn_target_match),
            "rpn_target_box": tensor_feature(rpn_target_box)
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        tfrec_writer.write(example.SerializeToString())
    tfrec_writer.close()


def parse_single_example(single_record):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "masks": tf.io.FixedLenFeature([], tf.string),
        "gt_boxes": tf.io.FixedLenFeature([], tf.string),
        "labels": tf.io.FixedLenFeature([], tf.string),
        "rpn_target_match": tf.io.FixedLenFeature([], tf.string),
        "rpn_target_box": tf.io.FixedLenFeature([], tf.string)
    }
    feature = tf.io.parse_single_example(single_record, feature_description)

    image = tf.io.parse_tensor(feature['image'], tf.float64)[0]
    masks = tf.io.parse_tensor(feature['masks'], tf.int8)[0]
    gt_boxes = tf.io.parse_tensor(feature['gt_boxes'], tf.float32)[0]
    labels = tf.io.parse_tensor(feature['labels'], tf.int8)[0]
    rpn_target_match = tf.io.parse_tensor(feature['rpn_target_match'], tf.float32)[0]
    rpn_target_box = tf.io.parse_tensor(feature['rpn_target_box'], tf.float32)[0]

    return image, masks, gt_boxes, labels, rpn_target_match, rpn_target_box


def parse_coco_segment_tfrecord(tfrec_path, repeat=1, shuffle_buffer=1000, batch=1):
    coco_tfrec_dataset = tf.data.TFRecordDataset(tfrec_path, num_parallel_reads=2)
    parse_data = coco_tfrec_dataset\
        .repeat(repeat)\
        .shuffle(shuffle_buffer)\
        .map(parse_single_example)\
        .batch(batch, drop_remainder=True)\
        .prefetch(10)

    return parse_data
