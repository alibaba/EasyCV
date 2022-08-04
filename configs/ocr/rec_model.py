_base_ = ['configs/base.py']

model = dict(
    type='OCRRecNet',
    backbone=dict(
        type='MobileNetV1Enhance',
        scale=0.5,
        last_conv_stride=[1,2],
        last_pool_type='avg'),
    neck=dict(
        type='SequenceEncoder',
        in_channels=512,
        encoder_type='svtr',
        dims=64,
        depth=2,
        hidden_dims=120,
        use_guide=True),
    head=dict(
        type='CTCHead',
        in_channels=64,
        fc_decay=0.00001),
    postprocess=dict(
        type='CTCLabelDecode',
        character_type='ch',
        character_dict_path='/nas/code/ocr/PaddleOCR2Pytorch-main/pytorchocr/utils/ppocr_keys_v1.txt',
        use_space_char=True
    ),
    pretrained='/root/code/ocr/paddle_to_torch_tools/paddle_weights/ch_ptocr_v3_rec_infer.pth'
)

test_pipeline = [
    dict(type='OCRResizeNorm', img_shape=(48,320)),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]