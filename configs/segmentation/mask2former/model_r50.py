norm_cfg=dict(type='BN', requires_grad=False)
model = dict(
    type = "Mask2Former",
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3, 4),
        frozen_stages=-1,
        norm_cfg=norm_cfg,
        norm_eval=False),
    head=dict(
        type="Mask2FormerHead",
        pixel_decoder=dict(
            input_stride=[4,8,16,32],
            input_channel=[256,512,1024,2048],
        ),
        transformer_decoder=dict(
            in_channels=256,
        ),
        num_classes=133,
    ),
    pretrained = "https://download.pytorch.org/models/resnet50-19c8e357.pth",
)