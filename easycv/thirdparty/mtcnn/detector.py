import torch
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw
from torch.autograd import Variable
from .get_nets import PNet, RNet, ONet
from .utils import (
    try_gpu,
    nms,
    calibrate_box,
    convert_to_square,
    correct_bboxes,
    get_image_boxes,
    generate_bboxes,
    preprocess,
)


class FaceDetector:
    def __init__(self, device=None, dir_path=None):
        if device is None:
          device=try_gpu()
        self.device = device

        if dir_path is not None:
            self.pnet = PNet(dir_path).to(device)
            self.rnet = RNet(dir_path).to(device)
            self.onet = ONet(dir_path).to(device)
            self.onet.eval()
        else:
            # LOAD MODELS
            self.pnet = PNet().to(device)
            self.rnet = RNet().to(device)
            self.onet = ONet().to(device)
            self.onet.eval()

    def detect(
        self,
        image,
        min_face_size=20.0,
        thresholds=[0.6, 0.7, 0.8],
        nms_thresholds=[0.7, 0.7, 0.7],
    ):
        """
        Arguments:
            image: an instance of PIL.Image.
            min_face_size: a float number.
            thresholds: a list of length 3.
            nms_thresholds: a list of length 3.

        Returns:
            two float numpy arrays of shapes [n_boxes, 5] and [n_boxes, 10],
            bounding boxes and facial landmarks.
        """

        # this detector only support RGB Image input !!!!!!!!! # todo: fix eas
        if type(image) == np.ndarray:
            image = Image.fromarray(image)
            #image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))

        # detector need Image Input

        # BUILD AN IMAGE PYRAMID
        width, height = image.size
        min_length = min(height, width)

        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)

        # scales for scaling the image
        scales = []

        # scales the image so that
        # minimum size that we can detect equals to
        # minimum face size that we want to detect
        m = min_detection_size / min_face_size
        min_length *= m

        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m * factor ** factor_count)
            min_length *= factor
            factor_count += 1

        # STAGE 1

        # it will be returned
        bounding_boxes = []

        # run P-Net on different scales
        for s in scales:
            boxes = self.__run_first_stage(image, scale=s, threshold=thresholds[0])
            bounding_boxes.append(boxes)

        # collect boxes (and offsets, and scores) from different scales
        bounding_boxes = [i for i in bounding_boxes if i is not None]
        bounding_boxes = np.vstack(bounding_boxes)

        keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
        bounding_boxes = bounding_boxes[keep]

        # use offsets predicted by pnet to transform bounding boxes
        bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
        # shape [n_boxes, 5]

        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # STAGE 2

        img_boxes = get_image_boxes(bounding_boxes, image, size=24)
        with torch.no_grad():
            img_boxes = Variable(torch.FloatTensor(img_boxes).to(self.device))
            output = self.rnet(img_boxes)
            offsets = output[0].cpu().data.numpy()  # shape [n_boxes, 4]
            probs = output[1].cpu().data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > thresholds[1])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]

        keep = nms(bounding_boxes, nms_thresholds[1])
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # STAGE 3

        img_boxes = get_image_boxes(bounding_boxes, image, size=48)
        if len(img_boxes) == 0:
            return [], []
        with torch.no_grad():
            img_boxes = Variable(torch.FloatTensor(img_boxes).to(self.device))
            output = self.onet(img_boxes)
            landmarks = output[0].cpu().data.numpy()  # shape [n_boxes, 10]
            offsets = output[1].cpu().data.numpy()  # shape [n_boxes, 4]
            probs = output[2].cpu().data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > thresholds[2])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]
        landmarks = landmarks[keep]

        # compute landmark points
        width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
        height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
        xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
        landmarks[:, 0:5] = (
            np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
        )
        landmarks[:, 5:10] = (
            np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]
        )

        bounding_boxes = calibrate_box(bounding_boxes, offsets)
        keep = nms(bounding_boxes, nms_thresholds[2], mode="min")
        bounding_boxes = bounding_boxes[keep]
        landmarks = landmarks[keep]

        # reshape [x1,x2,..x5,y1,..y5] to [[x1,y1],...[x5,y5]]
        landmarks = [np.array(ld).reshape((5,2), order="F") for ld in landmarks]

        return bounding_boxes, landmarks

    def safe_detect(self, image,
        min_face_size=20.0,
        thresholds=[0.6, 0.7, 0.8],
        nms_thresholds=[0.7, 0.7, 0.7],
        score_thresholds=0.90):
        try:
            bbox, ld = self.detect(image, min_face_size, thresholds, nms_thresholds)
            
            _bbox = []
            _ld = []

            for idx,_ in enumerate(bbox):
                if bbox[idx][-1] >= score_thresholds:
                    _bbox.append(bbox[idx])
                    _ld.append(ld[idx])

            return _bbox, _ld
        except:
            return [], [] 

    def draw_bboxes(self, image):
        """Draw bounding boxes and facial landmarks.

        Arguments:
            image: an instance of PIL.Image.

        Returns:
            an instance of PIL.Image.
        """

        bounding_boxes, facial_landmarks = self.detect(image)

        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)

        for b in bounding_boxes:
            draw.rectangle([(b[0], b[1]), (b[2], b[3])], outline="white")

        for p in facial_landmarks:
            for i in range(5):
                draw.ellipse(
                    [(p[i] - 1.0, p[i + 5] - 1.0), (p[i] + 1.0, p[i + 5] + 1.0)],
                    outline="blue",
                )

        return img_copy

    def crop_faces(self, image, size=112):
        """Crop all face images.

        Arguments:
            image: an instance of PIL.Image.
            size: the side length of output images.

        Returns:
            a list of PIL.Image instances
        """

        bounding_boxes, _ = self.detect(image)
        img_list = []

        # convert bboxes to square
        square_bboxes = convert_to_square(bounding_boxes)

        for b in square_bboxes:
            face_img = image.crop((b[0], b[1], b[2], b[3]))
            face_img = face_img.resize((size, size), Image.BILINEAR)
            img_list.append(face_img)
        return img_list

    def __run_first_stage(self, image, scale, threshold):
        """Run P-Net, generate bounding boxes, and do NMS.

        Arguments:
            image: an instance of PIL.Image.
            scale: a float number,
                scale width and height of the image by this number.
            threshold: a float number,
                threshold on the probability of a face when generating
                bounding boxes from predictions of the net.

        Returns:
            a float numpy array of shape [n_boxes, 9],
                bounding boxes with scores and offsets (4 + 1 + 4).
        """

        # scale the image and convert it to a float array
        width, height = image.size
        sw, sh = math.ceil(width * scale), math.ceil(height * scale)
        img = image.resize((sw, sh), Image.BILINEAR)
        img = np.asarray(img, "float32")

        with torch.no_grad():
            img = Variable(torch.FloatTensor(preprocess(img)).to(self.device))
            output = self.pnet(img)
            probs = output[1].cpu().data.numpy()[0, 1, :, :]
            offsets = output[0].cpu().data.numpy()
            # probs: probability of a face at each sliding window
            # offsets: transformations to true bounding boxes

        boxes = generate_bboxes(probs, offsets, scale, threshold)
        if len(boxes) == 0:
            return None

        keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
        return boxes[keep]
