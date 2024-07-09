import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformers import BertTokenizer, BertModel

def get_model_download(link, save_path):
    tokenizer = BertTokenizer.from_pretrained(link)
    model = BertModel.from_pretrained(link)
    
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)

def main():
    # get_model_download('bert-base-uncased', 
    #                    os.path.join('..', 'bert_base'))
    # get_model_download('bert-base-multilingual-uncased',
    #                    os.path.join('..', 'mbert'))
    # get_model_download('emilyalsentzer/Bio_ClinicalBERT',
    #                    os.path.join('..', 'cbert'))
    # get_model_download('kykim/bert-kor-base',
    #                    os.path.join('..', 'bert_kor_base'))
    
    # from transformers import AutoModel, AutoTokenizer, AutoConfig, RobertaTokenizerFast
    # model_path = os.path.join("..", "RoBERTa-base-PM-M3-Voc-distill-align", "RoBERTa-base-PM-M3-Voc-distill-align-hf")
    # config = AutoConfig.from_pretrained(
    #     model_path
    # )
    # model = AutoModel.from_pretrained(model_path, config=config)
    # tokenizer = RobertaTokenizerFast.from_pretrained(model_path, 
    #                                           add_prefix_space=False)
    
    # text = '상기 환아 3 times vomiting'
    # print(tokenizer.convert_ids_to_tokens(tokenizer.encode('상기 환아 3 times vomiting')))
    # inputs = tokenizer('상기 환아 3 times vomiting', return_tensors='pt')
    # # print(model(**inputs))
    
    from transformers import AutoModel, AutoTokenizer, RobertaTokenizer
    # model_path = 'allenai/biomed_roberta_base'
    model_path = 'StivenLancheros/mBERT-base-Biomedical-NER'
    
    model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
                       
    text = '상기 환아 3 times vomiting'
    print(tokenizer.encode('상기 환아 3 times vomiting'))
    print(tokenizer.convert_ids_to_tokens(tokenizer.encode('상기 환아 3 times vomiting')))
    
if __name__ == "__main__":
    main()