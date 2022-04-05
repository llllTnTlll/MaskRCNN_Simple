import numpy as np
import math


class Config(object):
    # 图像信息相关
    IMAGE_RESIZE_MODE = "square"    # 图像裁剪方式
    IMAGE_MIN_DIM = 800             # 图像短边长度
    IMAGE_MAX_DIM = 1024            # 图像长边长度
    IMAGE_CHANNEL_COUNT = 3         # 图像通道数
    IMAGE_MIN_SCALE = 0

    # FPN网络相关
    FPN_POOL_SIZE = [7, 7]                # ROIAlign后池化大小
    MASK_POOL_SIZE = [14, 14]
    TOP_DOWN_PYRAMID_SIZE = 256           # 上采样1*1卷积filter，使特征层能够正常Add
    FPN_FC_LAYERS_SIZE = 1024             # 类别预测全连接层数量
    FRATURE_STRIDES = [4, 8, 16, 32, 64]  # FPN网络各层缩放系数 1/n

    # RPN网络相关
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)        # 锚定框像素大小
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]                    # 锚定框宽高比例
    RPN_ANCHOR_STRIDE = 1                              # 锚定框生成步长
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])  # ProposalLayer中调整偏移量数量级参数
    RPN_NMS_THRESHOLD = 0.7                            # ProposalLayer中建议框的非极大抑制的值 iou低于该值将被剔除
    RPN_NMS_LIMIT = 6000                               # ProposalLayer中非极大抑制前取得的框的数量
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256                  # How many anchors per image to use for RPN training

    # DetectionTargetLayer相关
    TRAIN_ROIS_PER_IMAGE = 200                         # DT层输出的roi区域数量
    ROI_POSITIVE_RATIO = 0.33                          # 正样本比例
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    MASK_SHAPE = [28, 28]                              # outputmask
    # ROIs kept after non-maximum suppression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # DetectionLayer相关
    DETECTION_MAX_INSTANCES = 100                      # 一张图片中的最大实例数量
    DETECTION_NMS_THRESHOLD = 0.3                      # 预测结果iou低于该值将被剔除
    DETECTION_MIN_CONFIDENCE = 0.3                     # 预测结果置信度低于该值将被剔除

    # 训练相关
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 3             # 类别总数，1为背景
    USE_MINI_MASK = True            # 缩放mask以减少内存消耗
    MINI_MASK_SHAPE = (56, 56)      # 所使用mini_mask大小
    BACKBONE = 'resnet101'
    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when predicting
    TRAIN_BN = False  # Defaulting to False since batch size is often small
    GRADIENT_CLIP_NORM = 5.0        # Adam优化器参数
    # 损失的比重
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }
    LEARNING_RATE = 1e-5           # 学习率
    EPOCHS = 10                    # 学习轮数

    def __init__(self, base_on):
        assert base_on in ['yaml', 'coco']

        self.ANCHOR_PER_LOCATION = len(self.RPN_ANCHOR_RATIOS)
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT
        # self.TOTAL_BATCH = int(math.floor(img_num / self.BATCH_SIZE))

        # 通过裁剪方式求得图像shape
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM,
                                         self.IMAGE_CHANNEL_COUNT])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM,
                                         self.IMAGE_CHANNEL_COUNT])

        # 针对coco数据集的设置
        if base_on is 'coco':
            # TODO: 开始训练前重写
            self.COCO_ANNOTATION_FILE = ''                                    # coco数据集位置
            self.COCO_INCLUDE_CROWD = False                                   # 数据集是否存在重叠现象
            self.C0C0_INCLUDE_MASK = True                                     # 数据集中是否包含mask信息
            self.COCO_INCLUDE_KEYPOINT = False                                # 数据集中是否包含关键点信息
            # TODO: 自动均值计算
            self.IMAGE_MEAN = np.array([[[102.9801, 115.9465, 122.7717]]])    # 图像均值
            self.COCO_DATA_SIZE = -1                                          # 数据使用限


