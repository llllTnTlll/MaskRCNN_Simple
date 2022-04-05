import tensorflow as tf
from config import Config
from bbox_ops import norm_boxes_graph
from backbones import Resnet
from layers import ProposalLayer, DetectionTargetLayer, PyramidROIAlignLayer, DetectionLayer
import numpy as np
from losses import *


class MaskRCNN:
    def __init__(self, cfg: Config):
        self.config = cfg

    @staticmethod
    def rpn_graph(feature_map, anchors_per_location, anchor_stride, fpn_level):
        """ """
        # TODO: check if stride of 2 causes alignment issues if the feature map
        shared = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', strides=anchor_stride,
                                        name='rpn_conv_shared_' + fpn_level)(feature_map)
        x = tf.keras.layers.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                                   activation='linear',
                                   # name='rpn_class_raw_' + fpn_level)(shared)
                                   name='rpn_class_raw_' + fpn_level)(shared)

        # Reshape to [batch, anchors, 2]
        rpn_class_logits = tf.reshape(x, [tf.shape(x)[0], -1, 2])

        # Softmax on last dimension of BG/FG.
        rpn_probs = tf.keras.layers.Softmax(name="rpn_class_xxx_" + fpn_level)(rpn_class_logits)

        # Bounding box refinement. [batch, H, W, anchors per location * depth]
        # where depth is [x, y, log(w), log(h)]
        x = tf.keras.layers.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
                                   activation='linear',
                                   # name='rpn_bbox_pred_' + fpn_level)(shared)
                                   name='rpn_bbox_pred_' + fpn_level)(shared)

        # Reshape to [batch, anchors, 4]
        rpn_bbox_delta = tf.reshape(x, [tf.shape(x)[0], -1, 4])

        return [rpn_class_logits, rpn_probs, rpn_bbox_delta]

    def fpn_classify(self, rois, mrcnn_feature_maps, is_training):
        """ mask-rcnn 分类,边框预测层
        :param rois:
        :param mrcnn_feature_maps:
        :return:
        """
        pooled = PyramidROIAlignLayer(
            image_shape=self.config.IMAGE_SHAPE,
            batch_size=self.config.BATCH_SIZE,
            pool_shape=self.config.FPN_POOL_SIZE
        ).call(rois, mrcnn_feature_maps)

        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(filters=self.config.FPN_FC_LAYERS_SIZE,
                                   kernel_size=self.config.FPN_POOL_SIZE),
            name="mrcnn_class_conv1"
        )(pooled)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x, training=is_training)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.ReLU())(x)
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(filters=self.config.FPN_FC_LAYERS_SIZE,
                                   kernel_size=(1, 1)),
            name="mrcnn_class_conv2"
        )(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x, training=is_training)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.ReLU())(x)
        shared = tf.squeeze(tf.squeeze(x, axis=3), axis=2)

        # 类别预测
        mrcnn_class_logits = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.config.NUM_CLASSES),
            name='mrcnn_class_logits'
        )(shared)
        mrcnn_class = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Softmax(),
            name="mrcnn_class"
        )(mrcnn_class_logits)
        # 边框预测
        mrcnn_bbox = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.config.NUM_CLASSES * 4,
                                  activation='linear'),
            name='mrcnn_bbox_fc'
        )(shared)
        mrcnn_bbox = tf.reshape(
            mrcnn_bbox,
            (tf.shape(mrcnn_bbox)[0], tf.shape(mrcnn_bbox)[1], self.config.NUM_CLASSES, 4),
            name="mrcnn_bbox")

        return mrcnn_class_logits, mrcnn_class, mrcnn_bbox

    def fpn_mask_predict(self, rois, mrcnn_feature_maps, is_training):
        """ mask-rcnn mask预测层
        :param rois:
        :param mrcnn_feature_maps:
        :return:
        """
        pooled = PyramidROIAlignLayer(
            image_shape=self.config.IMAGE_SHAPE,
            batch_size=self.config.BATCH_SIZE,
            pool_shape=self.config.MASK_POOL_SIZE
        ).call(rois, mrcnn_feature_maps)

        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(filters=256,
                                   kernel_size=(3, 3),
                                   padding="same"),
            name="mrcnn_mask_conv1"
        )(pooled)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x, training=is_training)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(filters=256,
                                   kernel_size=(3, 3),
                                   padding="same"),
            name="mrcnn_mask_conv2"
        )(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x, training=is_training)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(filters=256,
                                   kernel_size=(3, 3),
                                   padding="same"),
            name="mrcnn_mask_conv3"
        )(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x, training=is_training)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(filters=256,
                                   kernel_size=(3, 3),
                                   padding="same"),
            name="mrcnn_mask_conv4"
        )(x)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x, training=is_training)
        x = tf.keras.layers.ReLU()(x)

        # [batch, num_rois,  pool_size*2, pool_size*2, channels]
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2DTranspose(filters=256,
                                            kernel_size=(2, 2),
                                            strides=2,
                                            activation="relu"),
            name="mrcnn_mask_deconv"
        )(x)

        # [batch, num_rois, pool_size*2, pool_size*2, num_classes]
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(filters=self.config.NUM_CLASSES,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   activation="sigmoid"),
            name="mrcnn_mask"
        )(x)
        return x

    def build_graph(self, is_training=True):

        input_images = tf.keras.layers.Input(shape=self.config.IMAGE_SHAPE, batch_size=self.config.BATCH_SIZE)

        if is_training:
            input_gt_boxes = tf.keras.layers.Input(
                shape=[None, 4], batch_size=self.config.BATCH_SIZE, name="input_gt_boxes", dtype=tf.float32)
            # 对box归一化
            gt_boxes = tf.keras.layers.Lambda(lambda x: norm_boxes_graph(x,
                                                                         self.config.IMAGE_SHAPE[0:2]))(input_gt_boxes)

            # rpn类别目标值以及box目标值
            # rpn_target_match = tf.keras.layers.Input(shape=[None], batch_size=self.batch_size, dtype=tf.float32)
            # rpn_target_box = tf.keras.layers.Input(shape=[None, 4], batch_size=self.batch_size, dtype=tf.float32)

            if self.config.USE_MINI_MASK:
                gt_masks = tf.keras.layers.Input(
                    shape=[self.config.MINI_MASK_SHAPE[0], self.config.MINI_MASK_SHAPE[1], None],
                    batch_size=self.config.BATCH_SIZE,
                    name="input_gt_masks",
                    dtype=tf.int8)
            else:
                gt_masks = tf.keras.layers.Input(
                    shape=[self.config.IMAGE_SHAPE[0], self.config.IMAGE_SHAPE[1], None],
                    batch_size=self.config.BATCH_SIZE,
                    name="input_gt_masks",
                    dtype=tf.int8)
            gt_classes = tf.keras.layers.Input([None], batch_size=self.config.BATCH_SIZE, dtype=tf.int8)
            all_anchors = tf.keras.layers.Input([None, 4], batch_size=self.config.BATCH_SIZE, name="input_anchors")
            anchors = all_anchors[0]

        else:
            all_anchors = tf.keras.layers.Input([None, 4], batch_size=self.config.BATCH_SIZE, name="input_anchors")
            anchors = all_anchors[0]

        # 前卷积, 拿到各个特征层
        backbone = Resnet()
        rpn_feature_maps, mrcnn_feature_maps = backbone.build_backbone(input_images=input_images,
                                                                       architecture=self.config.BACKBONE)

        layer_outputs = []
        # rpn对每个特征层提取边框和类别
        for i, p in enumerate(rpn_feature_maps):
            layer_outputs.append(self.rpn_graph(feature_map=p,
                                                anchors_per_location=self.config.ANCHOR_PER_LOCATION,
                                                anchor_stride=self.config.RPN_ANCHOR_STRIDE,
                                                fpn_level=str(i)))

        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox_delta"]
        outputs = list(zip(*layer_outputs))
        outputs = [tf.keras.layers.Concatenate(axis=1, name=n)(list(o)) for o, n in zip(outputs, output_names)]
        # print(outputs)
        rpn_class_logits, rpn_class, rpn_bbox_delta = outputs
        # rpn_class_logits, rpn_class, rpn_bbox_delta = layer_outputs[0]

        if is_training:
            # proposal对score排序和采样,再将预测的box delta映射到对应的anchor上
            rpn_rois = ProposalLayer(
                proposal_count=self.config.POST_NMS_ROIS_TRAINING,
                nms_threshold=self.config.RPN_NMS_THRESHOLD,
                rpn_bbox_std_dev=self.config.RPN_BBOX_STD_DEV,
                rpn_nms_limit=self.config.RPN_NMS_LIMIT,
                batch_size=self.config.BATCH_SIZE).call(rpn_class, rpn_bbox_delta, anchors)

            # detect target把proposal输出的roi和gt_box做偏差计算, 同时筛选出指定数量的样本和对应的目标, 作为损失计算用
            rois, mrcnn_target_class_ids, mrcnn_target_bbox, mrcnn_target_mask = DetectionTargetLayer(
                batch_size=self.config.BATCH_SIZE,
                train_rois_per_image=self.config.TRAIN_ROIS_PER_IMAGE,
                roi_positive_ratio=self.config.ROI_POSITIVE_RATIO,
                bbox_std_dev=self.config.BBOX_STD_DEV,
                use_mini_mask=self.config.USE_MINI_MASK,
                mask_shape=self.config.MASK_SHAPE
            ).call(rpn_rois, gt_classes, gt_boxes, gt_masks)

            # mask rcnn网络预测最终类别和边框
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.fpn_classify(rois, mrcnn_feature_maps, is_training)
            # mask rcnn网络预测最终mask
            mrcnn_mask = self.fpn_mask_predict(rois, mrcnn_feature_maps, is_training)

            # Model
            inputs = [input_images, input_gt_boxes, gt_classes, gt_masks, all_anchors]
            outputs = [rpn_class_logits, rpn_class, rpn_bbox_delta,
                       rois, mrcnn_target_class_ids, mrcnn_target_bbox, mrcnn_target_mask,
                       mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask]
            model = tf.keras.models.Model(inputs, outputs, name='mask_rcnn')
            return model

        else:
            # proposal对score排序和采样,再将预测的box delta映射到对应的anchor上
            rpn_rois = ProposalLayer(
                proposal_count=self.config.POST_NMS_ROIS_INFERENCE,
                nms_threshold=self.config.RPN_NMS_THRESHOLD,
                rpn_bbox_std_dev=self.config.RPN_BBOX_STD_DEV,
                rpn_nms_limit=self.config.RPN_NMS_LIMIT,
                batch_size=self.config.BATCH_SIZE).call(rpn_class, rpn_bbox_delta, anchors)

            # mask rcnn网络预测最终类别和边框
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.fpn_classify(rpn_rois, mrcnn_feature_maps, is_training)

            # 利用rpn输出的roi和mask rcnn输出的类别和边框得到最终的box和class
            # [batch, num_detections, (y1, x1, y2, x2, class_id, score)]
            detections = DetectionLayer(
                batch_size=self.config.BATCH_SIZE,
                bbox_std_dev=self.config.BBOX_STD_DEV,
                detection_max_instances=self.config.DETECTION_MAX_INSTANCES,
                detection_nms_thres=self.config.DETECTION_NMS_THRESHOLD,
                detection_min_confidence=self.config.DETECTION_MIN_CONFIDENCE
            ).call(rpn_rois, mrcnn_class, mrcnn_bbox,
                   window=np.array([0, 0, self.config.IMAGE_SHAPE[0], self.config.IMAGE_SHAPE[1]], dtype=np.float32))

            # mask rcnn网络预测最终mask
            mrcnn_mask = self.fpn_mask_predict(detections[..., :4], mrcnn_feature_maps, is_training)

            model = tf.keras.Model([input_images, all_anchors],
                                   [detections, mrcnn_class, mrcnn_bbox, mrcnn_mask],
                                   name='mask_rcnn')
            return model


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    config = Config(base_on='coco')
    model = MaskRCNN(config).build_graph(is_training=True)
    model.summary()
