import os
import sys
import copy
import json
import logging
import shutil
import tarfile
import tempfile
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformers import AutoTokenizer, AutoModel

class MLKD_Classification_Head(nn.Module):
    def __init__(self,
                 hidden_size=768,
                 dropout=0.1,
                 class_num=2,
                 act_fn='gelu'):
        super(MLKD_Classification_Head, self).__init__()

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, class_num)
        self.act_fn = nn.GELU() if act_fn=='gelu' else nn.ReLU()

    def forward(self, cls_token_embedding):
        x = self.dropout(cls_token_embedding)
        x = self.dense(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class MLKD_Model(nn.Module):
    def __init__(self,
                 model_path='kykim/bert-kor-base',):
        super(MLKD_Model, self).__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        
    def forward(self, input_ids, type_ids, att_mask):
        # embed : Embedding Layer output : size(batch, seq_len, 768)
        # hidden_states : each encoders output : size(layer_num+1, batch, seq_len, 768) # +1 is Embedding Layers
        # att_mterices : each multi-head attention's Q*K^T : size(layer_num, batch, head_num, seq_len, seq_len)
        # pooled_output : CLS_Token Embedding : size(batch, 768)
        # output : prediction vector : size(batch, class_num)
        output = self.bert(input_ids=input_ids,
                           attention_mask=att_mask,
                           token_type_ids=type_ids,
                           output_attentions=True,
                           output_hidden_states=True)
         
        last_hidden_state, pooled_output, hidden_states, att_matrices = output[:]
        hidden_states = torch.stack(hidden_states, dim=0)
        att_matrices = torch.stack(att_matrices, dim=0)
        
        return last_hidden_state, hidden_states, att_matrices, pooled_output


class MLKD_Model_for_Classification(nn.Module):
    def __init__(self,
                 model_path='kykim/bert-kor-base',
                 class_num=2):
        super(MLKD_Model_for_Classification, self).__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        # self.classifier = nn.Linear(768, class_num)
        self.classifier = MLKD_Classification_Head(hidden_size=768,
                                                   dropout=self.bert.config.hidden_dropout_prob,
                                                   class_num=class_num)
        
    def forward(self, input_ids, type_ids, att_mask):
        # embed : Embedding Layer output : size(batch, seq_len, 768)
        # hidden_states : each encoders output : size(layer_num+1, batch, seq_len, 768) # +1 is Embedding Layers
        # att_mterices : each multi-head attention's Q*K^T : size(layer_num, batch, head_num, seq_len, seq_len)
        # pooled_output : CLS_Token Embedding : size(batch, 768)
        # output : prediction vector : size(batch, class_num)
        output = self.bert(input_ids=input_ids,
                           attention_mask=att_mask,
                           token_type_ids=type_ids,
                           output_attentions=True,
                           output_hidden_states=True)
         
        last_hidden_state, pooled_output, hidden_states, att_matrices = output[:]
        hidden_states = torch.stack(hidden_states, dim=0)
        att_matrices = torch.stack(att_matrices, dim=0)
        
        output = self.classifier(pooled_output)
        logit = torch.softmax(output, dim=-1)
        return (logit, output), (hidden_states, att_matrices)

def main():
    pass

if __name__ == "__main__":
    main()