import random

import cv2

from easycv.predictors import TorchYoloXPredictor

colors = [[255, 0, 0], [255, 255, 0], [255, 255, 0], [0, 255, 255]
          ] + [[random.randint(0, 255) for _ in range(3)] for _ in range(2000)]


def plot_boxes(outputs, imgs, save_path=None, color=None, line_thickness=None):
    for i in range(len(imgs)):
        x = outputs[i]['detection_boxes']
        score = outputs[i]['detection_scores']
        id = outputs[i]['detection_classes']
        label = outputs[i]['detection_class_names']

        if x is not None and x.shape[0] != 0:
            # Plots one bounding box on image img
            tl = int(
                line_thickness or round(0.002 *
                                        (imgs[i].shape[0] + imgs[i].shape[1]) /
                                        2)) + 1  # line/font thickness
            # tl = int(line_thickness)

            for num in range(x.shape[0]):
                c1, c2 = (int(x[num][0]), int(x[num][1])), (int(x[num][2]),
                                                            int(x[num][3]))
                cv2.rectangle(
                    imgs[i],
                    c1,
                    c2,
                    colors[id[num]],
                    thickness=tl,
                    lineType=cv2.LINE_AA)

                tf = max(tl - 1, 1)  # font thickness
                t_size = cv2.getTextSize(
                    label[num], 0, fontScale=tl / 10, thickness=tf)[0]

                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                cv2.rectangle(imgs[i], c1, c2, colors[id[num]], -1,
                              cv2.LINE_AA)  # filled
                cv2.putText(
                    imgs[i],
                    label[num], (c1[0], c1[1] - 2),
                    0,
                    1, [225, 0, 255],
                    thickness=max(tf - 10, 2),
                    lineType=cv2.LINE_AA)
                cv2.putText(
                    imgs[i],
                    str(score[num]), (c1[0], c1[1] - 10),
                    0,
                    1, [225, 0, 255],
                    thickness=max(tf - 10, 2),
                    lineType=cv2.LINE_AA)

        else:
            print('Frame {} has no object!'.format(i + 150))
        cv2.imwrite(save_path + str(i + 150) + '.jpg', imgs[i])

    return


def main():
    # output_path = '/apsarapangu/disk6/xinyi.zxy/huogui/epoch_50_export.blade'
    # output_path = '/apsarapangu/disk6/xinyi.zxy/huogui/epoch_50_end2end.blade'
    output_path = '/apsarapangu/disk6/xinyi.zxy/huogui/epoch_50.pth'
    config_path = '/apsarapangu/disk6/xinyi.zxy/export_test/handdet_yolox_0426.py'
    detector = TorchYoloXPredictor(output_path)

    video_path = '/apsarapangu/disk6/xinyi.zxy/huogui/F62843B3C60B252E41AF9D0AD.mp4'

    # save_path = '/apsarapangu/disk6/xinyi.zxy/huogui/result_export/'
    # save_path = '/apsarapangu/disk6/xinyi.zxy/huogui/result_end2end/'
    save_path = '/apsarapangu/disk6/xinyi.zxy/huogui/result/'
    imgs = []
    cap = cv2.VideoCapture(video_path)

    for i in range(150, 191):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, img = cap.read()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)

    outputs = detector.predict(imgs)

    plot_boxes(outputs, imgs, save_path)


if __name__ == '__main__':
    main()
