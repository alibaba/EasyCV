import json

from mmcv.runner import DefaultOptimizerConstructor, get_dist_info

from .builder import OPTIMIZER_BUILDERS


def get_vit_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    """
    Calculate lr decay rate for different ViT blocks.
    Reference from https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py
    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.
    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if '.pos_embed' in name or '.patch_embed' in name:
        layer_id = 0
    elif '.blocks.' in name and '.residual.' not in name:
        layer_id = int(name[name.find('.blocks.'):].split('.')[2]) + 1

    scale = lr_decay_rate**(num_layers + 1 - layer_id)

    return layer_id, scale


@OPTIMIZER_BUILDERS.register_module()
class LayerDecayOptimizerConstructor(DefaultOptimizerConstructor):

    def add_params(self, params, module):
        """Add all parameters of module to the params list.
        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.
        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.

        Reference from https://github.com/ViTAE-Transformer/ViTDet/blob/main/mmcv_custom/layer_decay_optimizer_constructor.py
        Note: Currently, this optimizer constructor is built for ViTDet.
        """

        parameter_groups = {}
        print(self.paramwise_cfg)
        lr_decay_rate = self.paramwise_cfg.get('layer_decay_rate')
        num_layers = self.paramwise_cfg.get('num_layers')
        print('Build LayerDecayOptimizerConstructor %f - %d' %
              (lr_decay_rate, num_layers))
        lr = self.base_lr
        weight_decay = self.base_wd

        for name, param in module.named_parameters():

            if not param.requires_grad:
                continue  # frozen weights

            if 'backbone' in name and ('.norm' in name or '.pos_embed' in name
                                       or '.gn.' in name or '.ln.' in name):
                group_name = 'no_decay'
                this_weight_decay = 0.
            else:
                group_name = 'decay'
                this_weight_decay = weight_decay

            if name.startswith('backbone'):
                layer_id, scale = get_vit_lr_decay_rate(
                    name, lr_decay_rate=lr_decay_rate, num_layers=num_layers)
            else:
                layer_id, scale = -1, 1
            group_name = 'layer_%d_%s' % (layer_id, group_name)

            if group_name not in parameter_groups:

                parameter_groups[group_name] = {
                    'weight_decay': this_weight_decay,
                    'params': [],
                    'param_names': [],
                    'lr_scale': scale,
                    'group_name': group_name,
                    'lr': scale * lr,
                }

            parameter_groups[group_name]['params'].append(param)
            parameter_groups[group_name]['param_names'].append(name)

        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    'param_names': parameter_groups[key]['param_names'],
                    'lr_scale': parameter_groups[key]['lr_scale'],
                    'lr': parameter_groups[key]['lr'],
                    'weight_decay': parameter_groups[key]['weight_decay'],
                }
            print('Param groups = %s' % json.dumps(to_display, indent=2))

        params.extend(parameter_groups.values())
