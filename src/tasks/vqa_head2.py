# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn

from param import args
from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU

# Max length including <bos> and <eos>
MAX_VQA_LENGTH = 20


class VQAModel(nn.Module):
    def __init__(self, num_answers):
        super().__init__()

        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_VQA_LENGTH
        )
        hid_dim = self.lxrt_encoder.dim

        #  231 Alternative VQA Answer heads
        self.logit_fc = nn.Sequential(
             nn.Tanh(),
             nn.Dropout(0.5),
             nn.Linear(hid_dim, num_answers),
             nn.Tanh(),
             BertLayerNorm(num_answers, eps=1e-12),
             nn.Dropout(0.5),
             nn.Linear(num_answers, num_answers)
        )

        # can also change the loss function to be
        # nn.KLDivLoss, need to convert scores to log probs first
        # add nn.LogSoftmax(dim=-1)

    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        _, x = self.lxrt_encoder(sent, (feat, pos))
        logit = self.logit_fc(x)

        return logit


