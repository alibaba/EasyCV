_base_ = './mae_vit_base_patch16_8xb64_100e_lrdecay065_fintune.py'

# optimizer
update_interval = 2
optimizer_config = dict(update_interval=update_interval)
eff_batch_size = 64 * 8 * update_interval  # 1024
lr_decay = 0.75
optimizer = dict(
    type='AdamW',
    lr=1e-3 * eff_batch_size / 256,
    weight_decay=0.05,
    paramwise_options={
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'pos_embed': dict(weight_decay=0.),
        'patch_embed': dict(lr_mult=lr_decay**13),
        '\\.0\\.': dict(lr_mult=lr_decay**12),
        '\\.1\\.': dict(lr_mult=lr_decay**11),
        '\\.2\\.': dict(lr_mult=lr_decay**10),
        '\\.3\\.': dict(lr_mult=lr_decay**9),
        '\\.4\\.': dict(lr_mult=lr_decay**8),
        '\\.5\\.': dict(lr_mult=lr_decay**7),
        '\\.6\\.': dict(lr_mult=lr_decay**6),
        '\\.7\\.': dict(lr_mult=lr_decay**5),
        '\\.8\\.': dict(lr_mult=lr_decay**4),
        '\\.9\\.': dict(lr_mult=lr_decay**3),
        '\\.10\\.': dict(lr_mult=lr_decay**2),
        '\\.11\\.': dict(lr_mult=lr_decay**1),
        'head': dict(lr_mult=1.0)
    })
