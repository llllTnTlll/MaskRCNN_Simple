import json

from model.mask_rcnn import MaskRCNN
from model.config import Config
from model.anchors_ops import get_anchors
import cv2 as cv
import glob
import os
from json_serializer import RespondPkg, respond2json


class Operations:
    @staticmethod
    def predict(img_list, h5_path, config_path=None):
        # 加载权重
        config = Config()
        if config_path is not None:
            config.read_config(config_path)
        # 读取部分训练设置
        image_shape = config.IMAGE_SHAPE
        scales = config.ANCHOR_SCALES
        ratios = config.ANCHOR_RATIOS
        feature_strides = config.FEATURE_STRIDE
        anchor_stride = config.ANCHOR_STRIDE

        # 实例化模型
        mrcnn = MaskRCNN(is_training=False,
                         config=config)

        # 加载模型权重
        mrcnn.load_weights(h5_path, by_name=True)

        anchors = get_anchors(image_shape=image_shape,
                              scales=scales,
                              ratios=ratios,
                              feature_strides=feature_strides,
                              anchor_stride=anchor_stride)

        for i in range(len(img_list)):
            final_boxes, final_class_ids, final_scores, final_mask, pre_img = mrcnn.predict(img_list[i],
                                                                                            anchors,
                                                                                            draw_detect_res_figure=True)
            # 将检测结果序列化为json
            respond = RespondPkg(pkg_type='predict_data', result=[final_boxes, final_class_ids])
            json_respond = respond2json(respond)
            cv.imwrite('../data/tmp/{}.jpg'.format(i), pre_img)

            return json_respond


if __name__ == "__main__":
    img_dir = r"C:\Users\zhiyuan\Desktop\temp\test"
    paths = glob.glob(os.path.join(img_dir, '*.jpg'))
    imgs = []
    for path in paths:
        img = cv.imread(path)
        imgs.append(img)
    if len(imgs) > 0:
        Operations.predict(imgs, "../weights/mrcnn-epoch-95.h5", None)