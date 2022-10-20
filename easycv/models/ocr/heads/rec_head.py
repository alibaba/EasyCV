# Modified from https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.6/ppocr/modeling/heads
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from easycv.models.builder import HEADS
from ..necks.squence_encoder import Im2Seq, SequenceEncoder


class SAREncoder(nn.Module):
    """
    Args:
        enc_bi_rnn (bool): If True, use bidirectional RNN in encoder.
        enc_drop_rnn (float): Dropout probability of RNN layer in encoder.
        enc_gru (bool): If True, use GRU, else LSTM in encoder.
        d_model (int): Dim of channels from backbone.
        d_enc (int): Dim of encoder RNN layer.
        mask (bool): If True, mask padding in RNN sequence.
    """

    def __init__(self,
                 enc_bi_rnn=False,
                 enc_drop_rnn=0.1,
                 enc_gru=False,
                 d_model=512,
                 d_enc=512,
                 mask=True,
                 **kwargs):
        super().__init__()
        assert isinstance(enc_bi_rnn, bool)
        assert isinstance(enc_drop_rnn, (int, float))
        assert 0 <= enc_drop_rnn < 1.0
        assert isinstance(enc_gru, bool)
        assert isinstance(d_model, int)
        assert isinstance(d_enc, int)
        assert isinstance(mask, bool)

        self.enc_bi_rnn = enc_bi_rnn
        self.enc_drop_rnn = enc_drop_rnn
        self.mask = mask

        # LSTM Encoder
        kwargs = dict(
            input_size=d_model,
            hidden_size=d_enc,
            num_layers=2,
            batch_first=True,
            dropout=enc_drop_rnn,
            bidirectional=enc_bi_rnn)

        if enc_gru:
            self.rnn_encoder = nn.GRU(**kwargs)
        else:
            self.rnn_encoder = nn.LSTM(**kwargs)

        # global feature transformation
        encoder_rnn_out_size = d_enc * (int(enc_bi_rnn) + 1)
        self.linear = nn.Linear(encoder_rnn_out_size, encoder_rnn_out_size)

    def forward(self, feat, valid_ratios=None):

        h_feat = feat.shape[2]  # bsz c h w
        feat_v = F.max_pool2d(
            feat, kernel_size=(h_feat, 1), stride=1, padding=0)
        feat_v = feat_v.squeeze(2)  # bsz * C * W
        feat_v = feat_v.permute(0, 2, 1).contiguous()  # bsz * W * C
        holistic_feat = self.rnn_encoder(feat_v)[0]  # bsz * T * C

        if valid_ratios is not None:
            valid_hf = []
            T = holistic_feat.size(1)
            for i, valid_ratio in enumerate(valid_ratios):
                valid_step = min(T, math.ceil(T * valid_ratio)) - 1
                # for i in range(valid_ratios.size(0)):
                #     valid_step = torch.min(T, torch.ceil(T * valid_ratios[i])) - 1
                valid_hf.append(holistic_feat[i, valid_step, :])
            valid_hf = torch.stack(valid_hf, dim=0)
        else:
            valid_hf = holistic_feat[:, -1, :]  # bsz * C
        holistic_feat = self.linear(valid_hf)  # bsz * C

        return holistic_feat


class BaseDecoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def forward_train(self, feat, out_enc, targets, valid_ratios):
        raise NotImplementedError

    def forward_test(self, feat, out_enc, valid_ratios):
        raise NotImplementedError

    def forward(self,
                feat,
                out_enc,
                label=None,
                valid_ratios=None,
                train_mode=True):
        self.train_mode = train_mode

        if train_mode:
            return self.forward_train(feat, out_enc, label, valid_ratios)
        return self.forward_test(feat, out_enc, valid_ratios)


class ParallelSARDecoder(BaseDecoder):
    """
    Args:
        out_channels (int): Output class number.
        enc_bi_rnn (bool): If True, use bidirectional RNN in encoder.
        dec_bi_rnn (bool): If True, use bidirectional RNN in decoder.
        dec_drop_rnn (float): Dropout of RNN layer in decoder.
        dec_gru (bool): If True, use GRU, else LSTM in decoder.
        d_model (int): Dim of channels from backbone.
        d_enc (int): Dim of encoder RNN layer.
        d_k (int): Dim of channels of attention module.
        pred_dropout (float): Dropout probability of prediction layer.
        max_seq_len (int): Maximum sequence length for decoding.
        mask (bool): If True, mask padding in feature map.
        start_idx (int): Index of start token.
        padding_idx (int): Index of padding token.
        pred_concat (bool): If True, concat glimpse feature from
            attention with holistic feature and hidden state.
    """

    def __init__(
            self,
            out_channels,  # 90 + unknown + start + padding
            enc_bi_rnn=False,
            dec_bi_rnn=False,
            dec_drop_rnn=0.0,
            dec_gru=False,
            d_model=512,
            d_enc=512,
            d_k=64,
            pred_dropout=0.1,
            max_text_length=30,
            mask=True,
            pred_concat=True,
            **kwargs):
        super().__init__()

        self.num_classes = out_channels
        self.enc_bi_rnn = enc_bi_rnn
        self.d_k = d_k
        self.start_idx = out_channels - 2
        self.padding_idx = out_channels - 1
        self.max_seq_len = max_text_length
        self.mask = mask
        self.pred_concat = pred_concat

        encoder_rnn_out_size = d_enc * (int(enc_bi_rnn) + 1)
        decoder_rnn_out_size = encoder_rnn_out_size * (int(dec_bi_rnn) + 1)

        # 2D attention layer
        self.conv1x1_1 = nn.Linear(decoder_rnn_out_size, d_k)
        self.conv3x3_1 = nn.Conv2d(
            d_model, d_k, kernel_size=3, stride=1, padding=1)
        self.conv1x1_2 = nn.Linear(d_k, 1)

        # Decoder RNN layer

        kwargs = dict(
            input_size=encoder_rnn_out_size,
            hidden_size=encoder_rnn_out_size,
            num_layers=2,
            batch_first=True,
            dropout=dec_drop_rnn,
            bidirectional=dec_bi_rnn)
        if dec_gru:
            self.rnn_decoder = nn.GRU(**kwargs)
        else:
            self.rnn_decoder = nn.LSTM(**kwargs)

        # Decoder input embedding
        self.embedding = nn.Embedding(
            self.num_classes,
            encoder_rnn_out_size,
            padding_idx=self.padding_idx)

        # Prediction layer
        self.pred_dropout = nn.Dropout(pred_dropout)
        pred_num_classes = self.num_classes - 1
        if pred_concat:
            fc_in_channel = decoder_rnn_out_size + d_model + encoder_rnn_out_size
        else:
            fc_in_channel = d_model
        self.prediction = nn.Linear(fc_in_channel, pred_num_classes)

    def _2d_attention(self,
                      decoder_input,
                      feat,
                      holistic_feat,
                      valid_ratios=None):

        y = self.rnn_decoder(decoder_input)[0]
        # y: bsz * (seq_len + 1) * hidden_size

        attn_query = self.conv1x1_1(y)  # bsz * (seq_len + 1) * attn_size
        bsz, seq_len, attn_size = attn_query.shape
        attn_query = attn_query.view(bsz, seq_len, attn_size, 1, 1)
        # (bsz, seq_len + 1, attn_size, 1, 1)

        attn_key = self.conv3x3_1(feat)
        # bsz * attn_size * h * w
        attn_key = attn_key.unsqueeze(1)
        # bsz * 1 * attn_size * h * w

        attn_weight = torch.tanh(torch.add(attn_key, attn_query, alpha=1))

        # bsz * (seq_len + 1) * attn_size * h * w
        attn_weight = attn_weight.permute(0, 1, 3, 4, 2).contiguous()
        # bsz * (seq_len + 1) * h * w * attn_size
        attn_weight = self.conv1x1_2(attn_weight)
        # bsz * (seq_len + 1) * h * w * 1
        bsz, T, h, w, c = attn_weight.size()
        assert c == 1

        if valid_ratios is not None:
            # cal mask of attention weight
            attn_mask = torch.zeros_like(attn_weight)
            for i, valid_ratio in enumerate(valid_ratios):
                valid_width = min(w, math.ceil(w * valid_ratio))
                attn_mask[i, :, :, valid_width:, :] = 1
            attn_weight = attn_weight.masked_fill(attn_mask.bool(),
                                                  float('-inf'))
        # if valid_ratios is not None:
        #     # cal mask of attention weight
        #     for i in range(valid_ratios.size(0)):
        #         valid_width = torch.min(w, torch.ceil(w * valid_ratios[i]))
        #         # valid_width = paddle.minimum(
        #         #     w, paddle.ceil(valid_ratios[i] * w).astype("int32"))
        #         if valid_width < w:
        #             attn_weight[i, :, :, valid_width:, :] = float('-inf')

        attn_weight = attn_weight.view(bsz, T, -1)
        attn_weight = F.softmax(attn_weight, dim=-1)

        attn_weight = attn_weight.view(bsz, T, h, w,
                                       c).permute(0, 1, 4, 2, 3).contiguous()
        # attn_weight: bsz * T * c * h * w
        # feat: bsz * c * h * w
        attn_feat = torch.sum(
            torch.mul(feat.unsqueeze(1), attn_weight), (3, 4), keepdim=False)
        # bsz * (seq_len + 1) * C

        # Linear transformation
        if self.pred_concat:
            hf_c = holistic_feat.size(-1)
            holistic_feat = holistic_feat.expand(bsz, seq_len, hf_c)
            y = self.prediction(torch.cat((y, attn_feat, holistic_feat), 2))
        else:
            y = self.prediction(attn_feat)
        # bsz * (seq_len + 1) * num_classes
        if self.train_mode:
            y = self.pred_dropout(y)

        return y

    def forward_train(self, feat, out_enc, label, valid_ratios=None):

        lab_embedding = self.embedding(label)
        # bsz * seq_len * emb_dim
        out_enc = out_enc.unsqueeze(1)
        # bsz * 1 * emb_dim
        in_dec = torch.cat((out_enc, lab_embedding), dim=1)
        # bsz * (seq_len + 1) * C
        out_dec = self._2d_attention(
            in_dec, feat, out_enc, valid_ratios=valid_ratios)

        return out_dec[:, 1:, :]  # bsz * seq_len * num_classes

    def forward_test(self, feat, out_enc, valid_ratios=None):

        seq_len = self.max_seq_len
        bsz = feat.shape[0]
        start_token = torch.full((bsz, ),
                                 self.start_idx,
                                 device=feat.device,
                                 dtype=torch.long)
        # bsz
        start_token = self.embedding(start_token)
        # bsz * emb_dim
        emb_dim = start_token.shape[1]
        start_token = start_token.unsqueeze(1)
        start_token = start_token.unsqueeze(1).expand(-1, seq_len, -1)
        # bsz * seq_len * emb_dim
        out_enc = out_enc.unsqueeze(1)
        # bsz * 1 * emb_dim
        decoder_input = torch.cat((out_enc, start_token), dim=1)
        # bsz * (seq_len + 1) * emb_dim

        outputs = []
        for i in range(1, seq_len + 1):
            decoder_output = self._2d_attention(
                decoder_input, feat, out_enc, valid_ratios=valid_ratios)
            char_output = decoder_output[:, i, :]  # bsz * num_classes
            char_output = F.softmax(char_output, -1)
            outputs.append(char_output)
            _, max_idx = torch.max(char_output, dim=1, keepdim=False)
            char_embedding = self.embedding(max_idx)  # bsz * emb_dim
            if i < seq_len:
                decoder_input[:, i + 1, :] = char_embedding

        outputs = torch.stack(outputs, 1)  # bsz * seq_len * num_classes

        return outputs


@HEADS.register_module()
class SARHead(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 enc_dim=512,
                 max_text_length=30,
                 enc_bi_rnn=False,
                 enc_drop_rnn=0.1,
                 enc_gru=False,
                 dec_bi_rnn=False,
                 dec_drop_rnn=0.0,
                 dec_gru=False,
                 d_k=512,
                 pred_dropout=0.1,
                 pred_concat=True,
                 **kwargs):
        super(SARHead, self).__init__()

        # encoder module
        self.encoder = SAREncoder(
            enc_bi_rnn=enc_bi_rnn,
            enc_drop_rnn=enc_drop_rnn,
            enc_gru=enc_gru,
            d_model=in_channels,
            d_enc=enc_dim)

        # decoder module
        self.decoder = ParallelSARDecoder(
            out_channels=out_channels,
            enc_bi_rnn=enc_bi_rnn,
            dec_bi_rnn=dec_bi_rnn,
            dec_drop_rnn=dec_drop_rnn,
            dec_gru=dec_gru,
            d_model=in_channels,
            d_enc=enc_dim,
            d_k=d_k,
            pred_dropout=pred_dropout,
            max_text_length=max_text_length,
            pred_concat=pred_concat)

    def forward(self, feat, label, valid_ratios=None):
        '''
        img_metas: [label, valid_ratio]
        '''
        holistic_feat = self.encoder(feat, valid_ratios)  # bsz c

        if self.training:
            final_out = self.decoder(
                feat, holistic_feat, label, valid_ratios=valid_ratios)
        else:
            final_out = self.decoder(
                feat,
                holistic_feat,
                label=None,
                valid_ratios=valid_ratios,
                train_mode=False)

        return final_out


@HEADS.register_module()
class CTCHead(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels=6625,
                 fc_decay=0.0004,
                 mid_channels=None,
                 return_feats=False,
                 **kwargs):
        super(CTCHead, self).__init__()
        if mid_channels is None:
            self.fc = nn.Linear(
                in_channels,
                out_channels,
                bias=True,
            )
        else:
            self.fc1 = nn.Linear(
                in_channels,
                mid_channels,
                bias=True,
            )
            self.fc2 = nn.Linear(
                mid_channels,
                out_channels,
                bias=True,
            )

        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.return_feats = return_feats

    def forward(self, x, labels=None):
        if self.mid_channels is None:
            predicts = self.fc(x)
        else:
            x = self.fc1(x)
            predicts = self.fc2(x)

        if self.return_feats:
            result = (x, predicts)
        else:
            result = predicts

        if not self.training:
            predicts = F.softmax(predicts, dim=2)
            result = predicts

        return result


@HEADS.register_module()
class MultiHead(nn.Module):

    def __init__(self, in_channels, out_channels_list, **kwargs):
        super().__init__()
        self.head_list = kwargs.pop('head_list')
        head_name = [head.type for head in self.head_list]
        self.gtc_head = 'sar' if 'SARHead' in head_name else 'ctc'
        # assert len(self.head_list) >= 2
        for idx, head_name in enumerate(self.head_list):
            name = head_name.type
            if name == 'SARHead':
                # sar head
                sar_args = self.head_list[idx]
                self.sar_head = eval(name)(
                    in_channels=in_channels,
                    out_channels=out_channels_list['SARLabelDecode'],
                    **sar_args)
            elif name == 'CTCHead':
                # ctc neck
                self.encoder_reshape = Im2Seq(in_channels)
                neck_args = self.head_list[idx].Neck
                # encoder_type = neck_args.pop('type')
                encoder_type = neck_args.get('type')
                self.encoder = encoder_type
                self.ctc_encoder = SequenceEncoder(
                    in_channels=in_channels,
                    encoder_type=encoder_type,
                    **neck_args)
                # ctc head
                head_args = self.head_list[idx].Head
                self.ctc_head = eval(name)(
                    in_channels=self.ctc_encoder.out_channels,
                    out_channels=out_channels_list['CTCLabelDecode'],
                    **head_args)
            else:
                raise NotImplementedError(
                    '{} is not supported in MultiHead yet'.format(name))

    def forward(self, x, label=None, valid_ratios=None):
        ctc_encoder = self.ctc_encoder(x)
        ctc_out = self.ctc_head(ctc_encoder)
        head_out = dict()
        head_out['ctc'] = ctc_out
        head_out['ctc_neck'] = ctc_encoder
        # eval mode
        if not self.training:
            return ctc_out
        if self.gtc_head == 'sar':
            sar_out = self.sar_head(x, label, valid_ratios)
            head_out['sar'] = sar_out
            return head_out
        else:
            return head_out
