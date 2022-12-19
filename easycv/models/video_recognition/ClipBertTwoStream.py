# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
from torch import nn
import torch.nn.functional as F

from easycv.models import builder
from easycv.models.base import BaseModel
from easycv.models.builder import MODELS
from einops import rearrange


def load_state_dict_with_mismatch(model, loaded_state_dict_or_path):
    """operated in-place, no need to return `model`"""

    if isinstance(loaded_state_dict_or_path, str):
        loaded_state_dict = torch.load(
            loaded_state_dict_or_path, map_location="cpu")
    else:
        loaded_state_dict = loaded_state_dict_or_path
    model_keys = set([k for k in list(model.state_dict().keys())])
    load_keys = set(loaded_state_dict.keys())

    toload = {}
    mismatched_shape_keys = []
    for k in model_keys:
        k_rename = k.replace('encoder_text','encoder')
        k_rename = k_rename.replace('encoder_co','encoder')
        if k_rename in load_keys:
            if model.state_dict()[k].shape != loaded_state_dict[k_rename].shape:
                mismatched_shape_keys.append(k)
            else:
                toload[k] = loaded_state_dict[k_rename]
    import logging
    _LOG_FMT = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s'
    _DATE_FMT = '%m/%d/%Y %H:%M:%S'
    logging.basicConfig(format=_LOG_FMT, datefmt=_DATE_FMT, level=logging.INFO)
    LOGGER = logging.getLogger('__main__')  # this is the global logger
    LOGGER.info("You can ignore the keys with `num_batches_tracked` or from task heads")
    LOGGER.info("Keys in loaded but not in model:")
    diff_keys = load_keys.difference(model_keys)
    LOGGER.info(f"In total {len(diff_keys)}, {sorted(diff_keys)}")
    LOGGER.info("Keys in model but not in loaded:")
    diff_keys = model_keys.difference(load_keys)
    LOGGER.info(f"In total {len(diff_keys)}, {sorted(diff_keys)}")
    LOGGER.info("Keys in model and loaded, but shape mismatched:")
    LOGGER.info(f"In total {len(mismatched_shape_keys)}, {sorted(mismatched_shape_keys)}")
    model.load_state_dict(toload, strict=False)


@MODELS.register_module()
class ClipBertTwoStream(BaseModel):
    def __init__(
        self,
        vision,
        text,
        train_cfg=None,
        test_cfg=None,
        vison_pretrained=None,
        text_pretrained=None,
        loss_cls=dict(type='CrossEntropyLoss')
    ):
        super(ClipBertTwoStream, self).__init__()
        self.vision = builder.build_backbone(vision)
        self.text = builder.build_backbone(text)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        self.vison_pretrained = vison_pretrained
        self.text_pretrained = text_pretrained
        self.loss_cls = builder.build_loss(loss_cls)
        self.activate_fn = nn.Softmax(dim=1)
        
        self.init_weights()
        
    def init_weights(self):
        
        if self.vison_pretrained!=None:
            self.vision.init_weights(pretrained=self.vison_pretrained)
        else:
            self.vision.init_weights()
        if self.text_pretrained!=None:
            load_state_dict_with_mismatch(self.text, self.text_pretrained)
        else:
            self.text.init_weights()

 
    def extract_feat(self, imgs, text_input_ids, text_input_mask):
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        visual_feature = self.vision(imgs)
        visual_feature = rearrange(visual_feature, 'n c d h w -> n d h w c')
        logits = self.text(
            text_input_ids = text_input_ids,
            visual_inputs = visual_feature,
            text_input_mask = text_input_mask,
        )
        
        return logits
    
    def forward_train(self, imgs, text_input_ids, text_input_mask, label, **kwargs):
        """Defines the computation performed at every call when training."""
        losses = dict()

        cls_score = self.extract_feat(imgs, text_input_ids, text_input_mask)


        gt_labels = label.squeeze()
        loss_cls = self.loss_cls(cls_score, gt_labels)
        losses.update(loss_cls)

        return losses

    def forward_test(self, imgs, text_input_ids, text_input_mask, label=None, **kwargs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        cls_score = self.extract_feat(imgs, text_input_ids, text_input_mask)
        if label is not None:
            return dict(neck=cls_score.cpu(), label=label.cpu())
        else:
            result = {}
            result['prob'] = self.activate_fn(cls_score.cpu())
            result['class'] = torch.argmax(result['prob'])
            return result
        
    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the \
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        losses = self(**data, mode='train')
        loss, log_vars = self._parse_losses(losses)

        if type(data['imgs']) == list:
            num_samples = len(data['imgs'][0])
        else:
            num_samples = len(data['imgs'].data)

        return dict(loss=loss, log_vars=log_vars, num_samples=num_samples)
    
    
        
            
            
        
        
        
        
        
        