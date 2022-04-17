import numpy as np


def compute_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return cmin, rmin, cmax - cmin, rmax - rmin


def compute_iou(rec1, rec2):
    """
    计算两个矩形框的交并比。
    :param rec1: (x0,y0,x1,y1)      (x0,y0)代表矩形左上的顶点，（x1,y1）代表矩形右下的顶点。下同。
    :param rec2: (x0,y0,x1,y1)
    :return: 交并比IOU.
    """
    left_column_max = max(rec1[0], rec2[0])
    right_column_min = min(rec1[2], rec2[2])
    up_row_max = max(rec1[1], rec2[1])
    down_row_min = min(rec1[3], rec2[3])
    # 两矩形无相交区域的情况
    if left_column_max >= right_column_min or down_row_min <= up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        s1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        s2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
        s_cross = (down_row_min - up_row_max) * (right_column_min - left_column_max)
        return s_cross / (s1 + s2 - s_cross)


def wh2xy(box):
    x, y, w, h = box
    x0 = x - w/2
    y0 = y - h/2
    x1 = x + w/2
    y1 = y + h/2
    return x0, y0, x1, y1


if __name__ == "__main__":
    rec1 = wh2xy([10, 10, 10, 10])
    rec2 = wh2xy([20, 10, 20, 10])
    iou = compute_iou(rec1, rec2)
    print(iou)