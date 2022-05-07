_base_ = [
    '../_base_/models/fcn_r50-d8.py', 
    # '../_base_/datasets/pascal_voc12_aug.py',
    '../_base_/datasets/pascal_voc12.py',
    
]
model = dict(
    decode_head=dict(num_classes=21), auxiliary_head=dict(num_classes=21))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=True)

# runtime settings
# runner = dict(type='IterBasedRunner', max_iters=20000)
total_epochs = 100  # tmp epoch, need compute with max_iters

# checkpoint_config = dict(by_epoch=False, interval=2000)
# evaluation = dict(interval=2000, metric='mIoU', pre_eval=True)
checkpoint_config = dict(interval=1)
eval_config = dict(interval=1, gpu_collect=False)
#==========eval_pipelines is useless in current================
eval_pipelines = [
    dict(
        mode='test',
        evaluators=[dict(type='CocoDetectionEvaluator', classes=[])],
    )
]
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True

# disable opencv multithreading to avoid system being overloaded
# opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
# mp_start_method = 'fork'