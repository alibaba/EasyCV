import os
import unittest

import numpy as np
from PIL import Image
from tests.ut_config import TEST_IMAGES_DIR

from easycv.thirdparty.mtcnn import FaceDetector

bbox_res = [[
    1.06963833e+03, 5.70454030e+02, 1.53262074e+03, 1.17753027e+03,
    9.99988437e-01
],
            [
                1.64263477e+03, 7.14960351e+02, 1.99932316e+03, 1.17179306e+03,
                9.99982834e-01
            ],
            [
                4.89313601e+02, 6.55557247e+02, 8.37314858e+02, 1.12176724e+03,
                9.99867320e-01
            ]]


class DetDatasetTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_facedetector(self):
        detector = FaceDetector()
        image = Image.open(os.path.join(TEST_IMAGES_DIR, 'multi_face.jpg'))
        bboxes, landmarks = detector.detect(image)
        self.assertTrue(np.allclose(bboxes, np.array(bbox_res)))


if __name__ == '__main__':
    unittest.main()
