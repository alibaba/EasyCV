# Copyright (c) Alibaba, Inc. and its affiliates.


class CollateWrapper:

    def __init__(self, collate_fn, collate_hooks):
        from easycv.hooks.builder import build_hook

        self.collate_fn = collate_fn
        self.collate_hooks = [
            build_hook(hook_cfg) for hook_cfg in collate_hooks
        ]

    def __call__(self, batch):
        for hook in self.collate_hooks:
            batch = hook.before_collate(batch)
        batch = self.collate_fn(batch)
        for hook in self.collate_hooks:
            batch = hook.after_collate(batch)
        return batch
