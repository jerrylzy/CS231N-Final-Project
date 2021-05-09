# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn
import torch
from param import args
from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU
from lxrt.modeling import LXRTFeatureExtraction as VISUAL_CONFIG

# Max length including <bos> and <eos>
MAX_VQA_LENGTH = 20

class GatedTanh(nn.Module):
    def __init__(self, concat_dim, out_dim):
        super().__init__()
        self.w1 = nn.Linear(concat_dim, out_dim)
        self.w2 = nn.Linear(concat_dim, out_dim)

    def forward(self, x):
        # x shape: (b, o, concat_dim)
        y_hat = torch.tanh(self.w1(x))
        g = torch.sigmoid(self.w2(x))
        # (b, o, out_dim)
        return y_hat * g


class VQAModelAttn(nn.Module):
    def __init__(self, num_answers):
        super().__init__()

        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_VQA_LENGTH
        )
        hid_dim = self.lxrt_encoder.dim
        self.concat_dim = VISUAL_CONFIG.visual_feat_dim + hid_dim

        self.proj_a = GatedTanh(self.concat_dim, hid_dim)
        self.wa = nn.Linear(hid_dim, 1)
        self.proj_q = GatedTanh(hid_dim, hid_dim)
        self.proj_v = GatedTanh(VISUAL_CONFIG.visual_feat_dim, hid_dim)
        self.proj_h = GatedTanh(hid_dim, hid_dim)
        self.output_layer = nn.Linear(hid_dim, num_answers)

    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        b, o, f = feat.shape
        (_, visn_feats), pooled_output = self.lxrt_encoder(sent, (feat, pos))
        concat_feat = torch.cat((visn_feats, torch.repeat_interleave(pooled_output, o, 0).view(b, o, -1)), dim=-1)
        concat_feat = self.proj_a(concat_feat)
        attn_wgts = nn.Softmax(dim=1)(self.wa(concat_feat))
        weighted_v = (visn_feats.transpose(1, 2) @ attn_wgts).squeeze(2)
        h = self.proj_q(pooled_output) * self.proj_v(weighted_v)
        return self.output_layer(self.proj_h(h))
