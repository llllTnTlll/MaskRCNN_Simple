import numpy as np
import pickle


class Config:
    # 图像基本信息
    IMAGE_SHAPE = [512, 512, 3]

    # FPN网络相关
    FEATURE_STRIDE = [4, 8, 16, 32, 64]  # P2-P6特征层缩放系数 1/N

    # RPN网络相关
    ANCHOR_SCALES = [32, 64, 128, 256, 512]  # anchor像素边长
    ANCHOR_RATIOS = [0.5, 1, 2]  # anchor长宽比尺度
    ANCHOR_STRIDE = 1  # anchor生成步长 1为每像素生成len(ratios)个
    RPN_NMS_THRESHOLD = 0.7  # nms过滤的最低iou阈值
    RPN_NMS_LIMIT = 6000  # nms过滤输出数量限制
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])  # bbox归一化还原系数
    POST_NMS_ROIS_TRAINING = 2000  # 训练时筛选出的bbox数量
    POST_NMS_ROIS_INFERENCE = 1000  # 预测时筛选出的bbox数量

    # DetectionTargetLayer相关
    TRAIN_ROIS_PER_IMAGE = 200  # DT层过滤输出的roi数量
    ROI_POSITIVE_RATIO = 0.33  # DT层输出正样本比例

    # 训练相关
    EPOCHS = 100
    BATCH_SIZE = 1
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256  # 训练过程中经rpn网络提取的roi数量
    MASK_SHAPE = [28, 28]
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)
    TRAIN_WITH_TFRECORD = True

    # 验证相关
    CLASSES = ['_background_', 'rectangle', 'triangle', 'circle']
    FPN_POOL_SIZE = [7, 7]
    FPN_FC_LAYERS_SIZE = 1024
    MASK_POOL_SIZE = [14, 14]
    DETECTION_MIN_CONFIDENCE = 0.3
    DETECTION_NMS_THRES = 0.3
    DETECTION_MAX_INSTANCES = 100
    # TODO:自动计算mean_pixel
    PIXEL_MEAN = np.array([[[102.9801, 115.9465, 122.7717]]])

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

    def save_config(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)
        print("config file is successfully saved to: {}".format(file_path))

    def read_config(self, file_path):
        with open(file_path, 'rb') as file:
            config = pickle.load(file)
        for a in dir(config):
            if not a.startswith("__") and not callable(getattr(config, a)):
                setattr(self, a, getattr(config, a))

        print("config file was successfully read from: {}".format(file_path))

    def __init__(self):
        self.display()


if __name__ == "__main__":
    myconfig = Config()
    myconfig.EPOCHS = 33
    save_path = r'../config.txt'
    myconfig.save_config(save_path)
    new_config = Config()
    new_config.read_config(save_path)
    new_config.display()

