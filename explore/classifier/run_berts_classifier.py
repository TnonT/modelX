# -*- coding: utf-8 -*-

"""
@project: modelX
@author: heibai
@file: run_berts_classifier.py
@ide: PyCharm
@time 2021/7/7 10:39
"""
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertConfig, BertModel


class BertClassifier(nn.Module):
    def __init__(self, config, use_pool=False):
        super(BertClassifier, self).__init__()

        self.use_pool = use_pool
        self.num_classes = config.num_classes

        self.bert_encoder = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                                                           position_ids=None,
                                                           head_mask=None,
                                                           inputs_embeds=None,
                                                           labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output, pooled_output = outputs[0], outputs[1]

        if self.use_pool:
            output = pooled_output
        else:
            output = sequence_output[:, 0]

        output = self.dropout(output)
        logits = self.classifier(output)

        return logits


if __name__ == "__main__":

    data_path = ""













