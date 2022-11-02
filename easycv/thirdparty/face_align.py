import cv2
import numpy as np
from skimage import transform as trans
from PIL import Image


def glint360k_align(img, landmark, output_size=(112,112), pil_output=True):
    """
 
    Args:
        img: PIL Image or cv2 img(np array)
        landmark: faceid landmarks [[x1,y1],[x2,y2],..[x5,y5]] for [lefteye, righteye, nose, mouthleft, mouthright]
        output_size: output Image size
        pil_output: True return PIL Image, False return cv2 image

    Returns:
        A Image with PIL or cv2 format

    Raises:
        IOError: img is not PIL or cv2 Image
     
    """

    if type(img) is not np.ndarray:
        try:
            img = np.asarray(img)
        except:
            raise "alignment should input an ndarray or Image"

    dst = np.array([
   [30.2946, 51.6963],
   [65.5318, 51.5014],
   [48.0252, 71.7366],
   [33.5493, 92.3655],
   [62.7299, 92.2041]], dtype=np.float32 )
    dst[:,0] += 8.0
    src = landmark.astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(src, dst)
    M = tform.params[0:2, :]
    img = cv2.warpAffine(img, M, output_size, borderValue=0.0)

    if pil_output:
        img = Image.fromarray(img)

    return img