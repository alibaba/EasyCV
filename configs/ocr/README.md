# OCR algorithm
## PP-OCRv3
We convert [PaddleOCRv3](https://github.com/PaddlePaddle/PaddleOCR) models to pytorch style, and provide end2end interface to recognize text in images, by simplely load exported models.
### detection
We test on on icdar2015 dataset.
|Algorithm|backbone|configs|precison|recall|Hmean|Download|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|DB|MobileNetv3|[det_model_en.py](configs/ocr/detection/det_model_en.py)|0.7803|0.7250|0.7516|[log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/det/fintune_icdar2015_mobilev3/20220902_140307.log.json)-[model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/det/fintune_icdar2015_mobilev3/epoch_70.pth)|
|DB|R50|[det_model_en_r50.py](configs/ocr/detection/det_model_en_r50.py)|0.8622|0.8218|0.8415|[log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/det/fintune_icdar2015_r50/20220906_110252.log.json)-[model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/det/fintune_icdar2015_r50/epoch_1150.pth)|
### recognition
We test on on [DTRB](https://arxiv.org/abs/1904.01906) dataset.
|Algorithm|backbone|configs|acc|Download|
|:---:|:---:|:---:|:---:|:---:|
|SVTR|MobileNetv1|[rec_model_en.py](configs/ocr/recognition/rec_model_en.py)|0.7536|[log](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/rec/fintune_dtrb/20220914_125616.log.json)-[model](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/rec/fintune_dtrb/epoch_60.pth)|
### predict
We provide exported models contains weight and process config for easyly predict, which convert from PaddleOCRv3.
|Algorithm|Download|
|---|---|
|det|[ch_PP-OCRv3_det_sutdent](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/det/ch_PP-OCRv3_det/student_export.pth)|
||[Multilingual_PP-OCRv3_det_student](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/det/Multilingual_PP-OCRv3_det/student_export.pth)|
|rec|[ch_PP-OCRv3_rec_student](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/rec/ch_PP-OCRv3_rec/best_accuracy_student_export.pth)|
||[en_PP-OCRv3_rec_student](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/rec/en_PP-OCRv3_rec/best_accuracy.pth)|
|direction_cls|[ch_ppocr_mobile_v2.0_cls](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/cls/ch_ppocr_mobile_v2.0_cls/best_accuracy_export.pth)|
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
There are some ocr result.<br/>
![ocr_result1](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/result/test_ocr_1_out.jpg)
![ocr_result2](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/result/test_ocr_2_out.jpg)
