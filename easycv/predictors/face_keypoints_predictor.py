# Copyright (c) Alibaba, Inc. and its affiliates.

import cv2

from easycv.predictors.builder import PREDICTORS
from .base import OutputProcessor, PredictorV2

face_contour_point_index = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
]
left_eye_brow_point_index = [33, 34, 35, 36, 37, 38, 39, 40, 41, 33]
right_eye_brow_point_index = [42, 43, 44, 45, 46, 47, 48, 49, 50, 42]
left_eye_point_index = [66, 67, 68, 69, 70, 71, 72, 73, 66]
right_eye_point_index = [75, 76, 77, 78, 79, 80, 81, 82, 75]
nose_bridge_point_index = [51, 52, 53, 54]
nose_contour_point_index = [55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]
mouth_outer_point_index = [84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 84]
mouth_inter_point_index = [96, 97, 98, 99, 100, 101, 102, 103, 96]


class FaceKptsOutputProcessor(OutputProcessor):
    """Process the output of face keypoints models.

    Args:
        input_size (int): Target image size.
    """

    def __init__(self, input_size, point_number):
        self.input_size = input_size
        self.point_number = point_number

    def __call__(self, inputs):
        results = []

        img_metas = inputs['img_metas']
        points = inputs['point'].cpu().numpy()
        poses = inputs['pose'].cpu().numpy()

        for idx, point in enumerate(points):
            h, w, c = img_metas[idx]['img_shape']
            scale_h = h / self.input_size
            scale_w = w / self.input_size

            point = point.reshape((self.point_number, 2))
            for index in range(len(point)):
                point[index][0] *= scale_w
                point[index][1] *= scale_h

            results.append({'point': point, 'pose': poses[idx]})

        return results


@PREDICTORS.register_module()
class FaceKeypointsPredictor(PredictorV2):
    """Predict pipeline for face keypoint
    Args:
        model_path (str): Path of model path
        config_file (str): config file path for model and processor to init. Defaults to None.
        batch_size (int): batch size for forward.
        device (str): Support 'cuda' or 'cpu', if is None, detect device automatically.
        save_results (bool): Whether to save predict results.
        save_path (str): File path for saving results, only valid when `save_results` is True.
        pipelines (list[dict]): Data pipeline configs.
        input_processor_threads (int): Number of processes to process inputs.
        mode (str): The image mode into the model.
    """

    def __init__(
        self,
        model_path,
        config_file,
        batch_size=1,
        device=None,
        save_results=False,
        save_path=None,
        pipelines=None,
        input_processor_threads=8,
        mode='BGR',
    ):
        super(FaceKeypointsPredictor, self).__init__(
            model_path,
            config_file,
            batch_size=batch_size,
            device=device,
            save_results=save_results,
            save_path=save_path,
            pipelines=pipelines,
            input_processor_threads=input_processor_threads,
            mode=mode)

        self.input_size = self.cfg.IMAGE_SIZE
        self.point_number = self.cfg.POINT_NUMBER

    def model_forward(self, inputs):
        outputs = super().model_forward(inputs)
        outputs['img_metas'] = inputs['img_metas']
        return outputs

    def get_output_processor(self):
        return FaceKptsOutputProcessor(
            input_size=self.input_size, point_number=self.point_number)

    def show_result(self, img, points, scale=4.0, save_path=None):
        """Draw `result` over `img`.

        Args:
            img ( ndarray ): The image to be displayed.
            result (list): The face keypoints to draw over `img`.
            scale: zoom in or out scale
            save_path: path to save drawned 'img'
        Returns:
            img (ndarray): Only if not `show` or `out_file`
        """

        image = cv2.resize(img, dsize=None, fx=scale, fy=scale)

        def draw_line(point_index, image, point):
            for i in range(len(point_index) - 1):
                cur_index = point_index[i]
                next_index = point_index[i + 1]
                cur_pt = (int(point[cur_index][0] * scale),
                          int(point[cur_index][1] * scale))
                next_pt = (int(point[next_index][0] * scale),
                           int(point[next_index][1] * scale))
                cv2.line(image, cur_pt, next_pt, (0, 0, 255), thickness=2)

        draw_line(face_contour_point_index, image, points)
        draw_line(left_eye_brow_point_index, image, points)
        draw_line(right_eye_brow_point_index, image, points)
        draw_line(left_eye_point_index, image, points)
        draw_line(right_eye_point_index, image, points)
        draw_line(nose_bridge_point_index, image, points)
        draw_line(nose_contour_point_index, image, points)
        draw_line(mouth_outer_point_index, image, points)
        draw_line(mouth_inter_point_index, image, points)

        size = len(points)
        for i in range(size):
            x = int(points[i][0])
            y = int(points[i][1])
            cv2.putText(image, str(i), (int(x * scale), int(y * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.circle(image, (int(x * scale), int(y * scale)), 2, (0, 255, 0),
                       cv2.FILLED)

        if save_path is not None:
            cv2.imwrite(save_path, image)

        return image
