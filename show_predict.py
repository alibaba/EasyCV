import os
import random

import cv2
import numpy as np
from PIL import Image

from easycv.predictors.detector import TorchYoloXPredictor

colors = [[255, 0, 0], [255, 255, 0], [255, 255, 0], [0, 255, 255]
          ] + [[random.randint(0, 255) for _ in range(3)] for _ in range(2000)]


def plot_boxes(outputs, imgs, save_path=None, color=None, line_thickness=None):
    x = outputs['detection_boxes']
    score = outputs['detection_scores']
    id = outputs['detection_classes']
    label = outputs['detection_class_names']

    # Plots one bounding box on image img
    tl = int(
        line_thickness or round(0.002 * (imgs.shape[0] + imgs.shape[1]) /
                                2)) + 1  # line/font thickness
    # tl = int(line_thickness)

    for num in range(x.shape[0]):
        c1, c2 = (int(x[num][0]), int(x[num][1])), (int(x[num][2]),
                                                    int(x[num][3]))
        cv2.rectangle(
            imgs, c1, c2, colors[id[num]], thickness=tl, lineType=cv2.LINE_AA)

        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(
            label[num], 0, fontScale=tl / 10, thickness=tf)[0]

        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(imgs, c1, c2, colors[id[num]], -1, cv2.LINE_AA)  # filled
        cv2.putText(
            imgs,
            label[num], (c1[0], c1[1] - 2),
            0,
            0.2, [225, 0, 255],
            thickness=1,
            lineType=cv2.LINE_AA)
        cv2.putText(
            imgs,
            str(score[num]), (c1[0], c1[1] - 10),
            0,
            0.2, [225, 0, 255],
            thickness=1,
            lineType=cv2.LINE_AA)

    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_path + '/result_39.6.jpg', imgs)

    return


def main():
    pretrain_path = '/apsarapangu/disk5/zxy/pretrained/models/epoch_300_39.6.pth'
    data_path = '/apsarapangu/disk5/zxy/data/coco/'
    detection_model_path = pretrain_path

    img = os.path.join(data_path, 'val2017/000000037777.jpg')

    input_data_list = [np.asarray(Image.open(img))]
    predictor = TorchYoloXPredictor(
        model_path=detection_model_path, score_thresh=0.5)

    output = predictor.predict(input_data_list)[0]

    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plot_boxes(output, img, save_path='./result')
    print(output)


if __name__ == '__main__':
    main()
