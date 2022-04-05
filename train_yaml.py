from anchor_ops import get_anchors
from layers import build_rpn_targets
from config import Config
import numpy as np
from losses import *
from maskrcnn import MaskRCNN
from data_ops import ShapesDataset, data_generator
import os


class ShapesConfig(Config):
    NAME = "shapes"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 3
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512


# 禁用动态图结构
tf.compat.v1.disable_eager_execution()
# 获取训练文件根目录
dataset_root_path = r"C:\Users\zhiyuan\Downloads\temp"
img_floder = dataset_root_path + r"\imgs"
mask_floder = dataset_root_path + r"\mask"
yaml_floder = dataset_root_path + r"\yaml"
imglist = os.listdir(img_floder)

count = len(imglist)
np.random.seed(10101)
np.random.shuffle(imglist)

# 生成配置文件
config = ShapesConfig(img_num=count)

# 取前百分之九十为训练集
train_imglist = imglist[:int(count * 0.9)]
val_imglist = imglist[int(count * 0.9):]

# 训练数据集准备
dataset_train = ShapesDataset()
dataset_train.load_shapes(len(train_imglist), img_floder, mask_floder, train_imglist, yaml_floder)
dataset_train.prepare()

# 验证数据集准备
dataset_val = ShapesDataset()
dataset_val.load_shapes(len(val_imglist), img_floder, mask_floder, val_imglist, yaml_floder)
dataset_val.prepare()

# 生成数据集
train_generator = data_generator(dataset_train, config, shuffle=True,
                                 batch_size=config.BATCH_SIZE)
val_generator = data_generator(dataset_val, config, shuffle=True,
                               batch_size=config.BATCH_SIZE)


mrcnn = MaskRCNN(cfg=config).build_graph(is_training=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
anchors = get_anchors(image_shape=config.IMAGE_SHAPE,
                      scales=config.RPN_ANCHOR_SCALES,
                      ratios=config.RPN_ANCHOR_RATIOS,
                      feature_strides=config.FRATURE_STRIDES,
                      anchor_stride=config.RPN_ANCHOR_STRIDE)
all_anchors = np.stack([anchors, anchors], axis=0)

# tensorboard 日志目录
log_dir = './logs'
summary_writer = tf.summary.create_file_writer(log_dir)

for epoch in range(config.EPOCHS):
    for batch in range(config.TOTAL_BATCH):
        imgs, _, rpn_target_match, rpn_target_box, labels, gt_boxes, masks = next(train_generator)
        # rpn_target_match, rpn_target_box = build_rpn_targets(
        #     anchors=anchors,
        #     gt_boxes=gt_boxes,
        #     image_shape=config.IMAGE_SHAPE[0:2],
        #     batch_size=config.BATCH_SIZE,
        #     rpn_train_anchors_per_image=config.RPN_TRAIN_ANCHORS_PER_IMAGE,
        #     rpn_bbox_std_dev=config.RPN_BBOX_STD_DEV)

        print(np.shape(imgs))
        print(np.shape(masks))
        print(np.shape(gt_boxes))
        print("-------{}-{}--------".format(epoch, batch))

        if np.sum(gt_boxes) <= 0.:
            print(batch, " gt_boxes: ", gt_boxes)
            continue

        if epoch % 5 == 0 and epoch != 0:
            mrcnn.save_weights("./mrcnn-epoch-{}.h5".format(epoch))

        with tf.GradientTape() as tape:
            # 模型输出
            # rpn_target_match, rpn_target_box, rpn_class_logits, rpn_class, rpn_bbox_delta, rois, \
            rpn_class_logits, rpn_class, rpn_bbox_delta, rois, \
            mrcnn_target_class_ids, mrcnn_target_bbox, mrcnn_target_mask, mrcnn_class_logits, \
            mrcnn_class, mrcnn_bbox, mrcnn_mask = \
                mrcnn([imgs, gt_boxes, labels, masks, all_anchors], training=True)
            # mrcnn([imgs, gt_boxes, labels, masks, all_anchors], training=True)

            # 计算损失
            rpn_c_loss = rpn_class_loss(rpn_target_match, rpn_class_logits)
            rpn_b_loss = rpn_bbox_loss(rpn_target_box, rpn_target_match, rpn_bbox_delta)
            mrcnn_c_loss = mrcnn_class_loss(mrcnn_target_class_ids, mrcnn_class_logits, rois)
            mrcnn_b_loss = mrcnn_bbox_loss(mrcnn_target_bbox, mrcnn_target_class_ids, mrcnn_bbox, rois)
            mrcnn_m_bc_loss = mrcnn_mask_loss(mrcnn_target_mask, mrcnn_target_class_ids, mrcnn_mask, rois)
            total_loss = rpn_c_loss + rpn_b_loss + mrcnn_c_loss + mrcnn_b_loss + mrcnn_m_bc_loss

            # 梯度更新
            grad = tape.gradient(total_loss, mrcnn.trainable_variables)
            optimizer.apply_gradients(zip(grad, mrcnn.trainable_variables))

            # tensorboard 损失曲线
            with summary_writer.as_default():
                tf.summary.scalar('loss/rpn_class_loss', rpn_c_loss,
                                  step=epoch * config.TOTAL_BATCH + batch)
                tf.summary.scalar('loss/rpn_bbox_loss', rpn_b_loss,
                                  step=epoch * config.TOTAL_BATCH + batch)
                tf.summary.scalar('loss/mrcnn_class_loss', mrcnn_c_loss,
                                  step=epoch * config.TOTAL_BATCH + batch)
                tf.summary.scalar('loss/mrcnn_bbox_loss', mrcnn_b_loss,
                                  step=epoch * config.TOTAL_BATCH + batch)
                tf.summary.scalar('loss/mrcnn_mask_binary_crossentropy_loss', mrcnn_m_bc_loss,
                                  step=epoch * config.TOTAL_BATCH + batch)
                tf.summary.scalar('loss/total_loss', total_loss,
                                  step=epoch * config.TOTAL_BATCH + batch)
