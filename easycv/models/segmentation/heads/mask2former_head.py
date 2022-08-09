from torch import nn

from easycv.models.builder import HEADS
from .pixel_decoder import MSDeformAttnPixelDecoder
from .transformer_decoder import MultiScaleMaskedTransformerDecoder


@HEADS.register_module()
class Mask2FormerHead(nn.Module):

    def __init__(
        self,
        pixel_decoder,
        transformer_decoder,
        num_things_classes: int,
        num_stuff_classes: int,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        # extra parameters
        transformer_in_feature: str = 'multi_scale_pixel_decoder',
    ):
        """

        Args:
            pixel_decoder (cfg): config to build pixel decoder
            transformer_decoder (cfg): config to build transformer decoder
            num_things_classes (int): number of things classes
            num_stuff_classes (int): number of stuff classes
            loss_weight (float, optional): loss weight. Defaults to 1.0.
            ignore_value (int, optional): category id to be ignored during training. Defaults to -1.
            transformer_in_feature (str, optional): nput feature name to the transformer_predictor, only support multi_scale_pixel_decoder now. Defaults to 'multi_scale_pixel_decoder'.
        """
        super().__init__()

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight
        self.pixel_decoder = MSDeformAttnPixelDecoder(**pixel_decoder)
        self.predictor = MultiScaleMaskedTransformerDecoder(
            **transformer_decoder)
        self.transformer_in_feature = transformer_in_feature

        self.num_classes = num_things_classes + num_stuff_classes

    def forward(self, features, mask=None):

        return self.layers(features, mask)

    def layers(self, features, mask=None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
            features)
        if self.transformer_in_feature == 'multi_scale_pixel_decoder':
            predictions = self.predictor(multi_scale_features, mask_features,
                                         mask)
        else:
            if self.transformer_in_feature == 'transformer_encoder':
                assert (transformer_encoder_features is not None
                        ), 'Please use the TransformerEncoderPixelDecoder.'
                predictions = self.predictor(transformer_encoder_features,
                                             mask_features, mask)
            elif self.transformer_in_feature == 'pixel_embedding':
                predictions = self.predictor(mask_features, mask_features,
                                             mask)
            else:
                predictions = self.predictor(
                    features[self.transformer_in_feature], mask_features, mask)
        return predictions
