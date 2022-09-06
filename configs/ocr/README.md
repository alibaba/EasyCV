# OCR algorithm
## PP-OCRv3
&ensp;&ensp;We convert [PaddleOCRv3](https://github.com/PaddlePaddle/PaddleOCR) models to pytorch style, and fintuned on icdar2015 dataset. Futhermore, we provide end2end interface to recognize text in images, by simplely load exported models.
### detection
|Algorithm|backbone|configs|precison|recall|Hmean|Download|
|---|---|---|---|---|---|---|
|DB|MobileNetv3|[det_model_en.py](configs/ocr/det_model_en.py)|0.7803|0.7250|0.7516|[log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/det/fintune_icdar2015_mobilev3/20220902_140307.log.json)-[model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/det/fintune_icdar2015_mobilev3/epoch_70.pth)|
|DB|R50|[det_model_en_r50.py](configs/ocr/det_model_en_r50.py)|0.8622|0.8218|0.8415|[log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/det/fintune_icdar2015_r50/20220906_110252.log.json)-[model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/det/fintune_icdar2015_r50/epoch_1150.pth)|
### recognition
### predict
&ensp;&ensp;We provide exported models contains weight and process config for easyly predict, which convert from PaddleOCRv3.
|Algorithm|Download|
|---|---|
|det|[ch_PP-OCRv3_det_sutdent](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/det/ch_PP-OCRv3_det/student_export.pth)|
||[Multilingual_PP-OCRv3_det_student](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/det/Multilingual_PP-OCRv3_det/student_export.pth)|
|rec|[ch_PP-OCRv3_rec_student](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/rec/ch_PP-OCRv3_rec/best_accuracy_student_export.pth)|
||[en_PP-OCRv3_rec_student](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/rec/en_PP-OCRv3_rec/best_accuracy.pth)|
|direction_cls|[ch_ppocr_mobile_v2.0_cls](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/cls/ch_ppocr_mobile_v2.0_cls/best_accuracy.pth)|
```
import cv2
from easycv.predictors.ocr import OCRPredictor
! wget http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/simfang.ttf
! wget http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/ocr_det.jpg
predictor = OCRPredictor(
    det_model_path=PRETRAINED_MODEL_OCRDET,
    rec_model_path=PRETRAINED_MODEL_OCRREC,
    cls_model_path=PRETRAINED_MODEL_OCRCLS,
    use_angle_cls=True)
img = cv2.imread('ocr_det.jpg')
filter_boxes, filter_rec_res = predictor.predict_single(img)
out_img = predictor.show(
    filter_boxes,
    filter_rec_res,
    img,
    font_path='simfang.ttf')
cv2.imwrite('out_img.jpg', out_img)
```
