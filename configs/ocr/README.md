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
We provide exported models contain weights and process config for easyly predict, which convert from PaddleOCRv3.
#### det model
|language|Download|
|---|---|
|chinese|[chinese_det.pth](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/export_model/det/chinese_det.pth)|
|english|[english_det.pth](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/export_model/det/english_det.pth)|
|multilingual|[multilingual_det.pth](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/export_model/det/multilingual_det.pth)|
#### rec model
|language|Download|
|---|---|
|chiese|[chinese_rec.pth](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/export_model/rec/chinese_rec.pth)|
|english|[english_rec.pth](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/export_model/rec/english_rec.pth)|
|korean|[korean_rec.pth](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/export_model/rec/korean_rec.pth)|
|japan|[japan_rec.pth](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/export_model/rec/japan_rec.pth)|
|chinese_cht|[chinese_cht_rec.pth](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/export_model/rec/chinese_cht_rec.pth)|
|Telugu|[te_rec.pth](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/export_model/rec/te_rec.pth)|
|Canada|[ka_rec.pth](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/export_model/rec/ka_rec.pth)|
|Tamil|[ta_rec.pth](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/export_model/rec/ta_rec.pth)|
|latin|[latin_rec.pth](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/export_model/rec/latin_rec.pth)|
|cyrillic|[cyrillic_rec.pth](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/export_model/rec/cyrillic_rec.pth)|
|devanagari|[devanagari_rec.pth](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/export_model/rec/devanagari_rec.pth)|
#### direction model
|language|Download|
|---|---|
||[direction.pth](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/export_model/cls/direction.pth)|
#### usage
##### detection
```
import cv2
from easycv.predictors.ocr import OCRDetPredictor
img = cv2.imread(img_path)
predictor = OCRDetPredictor(model_path)
out = predictor([img])
out_img = predictor.show_result(out[0], img)
cv2.imwrite(out_img_path,out_img)
```
![det_result](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/result/det_result.jpg)
##### recognition
```
import cv2
from easycv.predictors.ocr import OCRRecPredictor
img = cv2.imread(img_path)
predictor = OCRRecPredictor(model_path)
out = predictor([img])
print(out)
```
![rec_input](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/export_model/test_image/japan_rec.jpg)<br/>
![rec_putput](http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/export_model/test_image/japan_predict.jpg)
##### end2end
```
import cv2
from easycv.predictors.ocr import OCRPredictor
! wget http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/simfang.ttf
! wget http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/ocr_det.jpg
predictor = OCRPredictor(
    det_model_path=path_to_detmodel,
    rec_model_path=path_to_recmodel,
    cls_model_path=path_to_clsmodel,
    use_angle_cls=True)
img = cv2.imread('ocr_det.jpg')
filter_boxes, filter_rec_res = predictor(img)
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
