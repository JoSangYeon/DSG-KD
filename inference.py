import os
import glob
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler

from sklearn.metrics import (roc_auc_score, 
                             average_precision_score,
                             accuracy_score,  
                             recall_score,
                             precision_score,
                             f1_score,
                             brier_score_loss, # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html
                             confusion_matrix,)

from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification
# from transformers import AdamW

from dataset import *
from learning import *
from model import *
from utils import *


def base_inference(model_init_path, ckpt_path, device, inference_data, tokenizer,
                   label_frequency=0.4503649727377383, sample_rate=0.75, bootstrap_K=10,
                   max_length=512, padding='max_length', class_num=2, batch_size=64,
                   SEED=17):
    METRIC_DICT = {
        'best_F1_model': {
            'loss': [],
            'acc': [],
            'TH_ACC': [],
            'AUROC': [],
            'AUPRC': [],
            'RECALL': [],
            'PRECISION': [],
            'F1': [],
            'BRIER': [],
        },
        'best_BRIER_model': {
            'loss': [],
            'acc': [],
            'TH_ACC': [],
            'AUROC': [],
            'AUPRC': [],
            'RECALL': [],
            'PRECISION': [],
            'F1': [],
            'BRIER': [],
        },
        'best_Loss_model': {
            'loss': [],
            'acc': [],
            'TH_ACC': [],
            'AUROC': [],
            'AUPRC': [],
            'RECALL': [],
            'PRECISION': [],
            'F1': [],
            'BRIER': [],
        },
    }

    METRIC_RESULT = {
        'loss': [],
        'acc': [],
        'TH_ACC': [],
        'AUROC': [],
        'AUPRC': [],
        'RECALL': [],
        'PRECISION': [],
        'F1': [],
        'BRIER': []
    }

    model = BertForSequenceClassification.from_pretrained(model_init_path, num_labels=class_num)
    test_list = ['best_F1_model', 'best_BRIER_model', 'best_Loss_model']

    for t_name in test_list:
        print(f'\t>>> {t_name} : ')
        checkpoint = torch.load(os.path.join(ckpt_path, t_name + '.tar'), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict']);
        model.to(device)

        for k in range(bootstrap_K):
            bootstrap_data = inference_data.sample(frac=sample_rate, random_state=SEED + k)
            bootstrap_data = bootstrap_data.reset_index(drop=True)
            bootstrap_dataset = Base_Dataset(bootstrap_data,
                                             tokenizer,
                                             max_length=max_length,
                                             padding=padding,
                                             return_tensors='pt',
                                             class_num=class_num)
            bootstrap_loader = DataLoader(bootstrap_dataset, batch_size=batch_size,
                                          sampler=RandomSampler(bootstrap_dataset))

            loss, acc, (TH_ACC, AUROC, AUPRC, RECALL, PRECISION, F1, BRIER) = base_evaluate(model,
                                                                                            device,
                                                                                            bootstrap_loader,
                                                                                            label_frequency)
            print(f'\t\tResult : loss:{loss:.4f}, AUPRC:{AUPRC:.4f}, AUROC:{AUROC:.4f}, RECALL:{RECALL:.4f}, PRECISION:{PRECISION:.4f}, BRIER:{BRIER:.4f}')
            METRIC_DICT[t_name]['loss'].append(loss)
            METRIC_DICT[t_name]['acc'].append(acc)
            METRIC_DICT[t_name]['TH_ACC'].append(TH_ACC)
            METRIC_DICT[t_name]['AUROC'].append(AUROC)
            METRIC_DICT[t_name]['AUPRC'].append(AUPRC)
            METRIC_DICT[t_name]['RECALL'].append(RECALL)
            METRIC_DICT[t_name]['PRECISION'].append(PRECISION)
            METRIC_DICT[t_name]['F1'].append(F1)
            METRIC_DICT[t_name]['BRIER'].append(BRIER)

    for key in METRIC_DICT.keys():
        models_metric = METRIC_DICT[key]
        for k in models_metric.keys():
            mean = np.round(np.mean(models_metric[k]), 4)
            std = np.round(np.std(models_metric[k]), 3)

            METRIC_RESULT[k].append(f"{mean}±{std}")
    print(METRIC_RESULT)
    # with open(os.path.join(ckpt_path, 'inference_result.json'),'w') as f:
    #     json.dump(METRIC_DICT, f, indent=4)

    model_result_df = pd.DataFrame(METRIC_RESULT, index=METRIC_DICT.keys())
    model_result_df.to_csv(os.path.join(ckpt_path, 'result.csv'))
    return model_result_df

def MLKD_inference(student_init_path, t_tokenizer, s_tokenizer,
                   ckpt_path, device, inference_data, label_frequency=0.4503649727377383,
                   sample_rate=0.75, bootstrap_K=10, max_length=512, padding='max_length',
                   class_num=2, batch_size=64, SEED=17):
    METRIC_DICT = {
        'best_F1_model': {
            'loss': [],
            'acc': [],
            'TH_ACC': [],
            'AUROC': [],
            'AUPRC': [],
            'RECALL': [],
            'PRECISION': [],
            'F1': [],
            'BRIER': [],
        },
        'best_BRIER_model': {
            'loss': [],
            'acc': [],
            'TH_ACC': [],
            'AUROC': [],
            'AUPRC': [],
            'RECALL': [],
            'PRECISION': [],
            'F1': [],
            'BRIER': [],
        },
        'best_Loss_model': {
            'loss': [],
            'acc': [],
            'TH_ACC': [],
            'AUROC': [],
            'AUPRC': [],
            'RECALL': [],
            'PRECISION': [],
            'F1': [],
            'BRIER': [],
        },
    }

    METRIC_RESULT = {
        'loss': [],
        'acc': [],
        'TH_ACC': [],
        'AUROC': [],
        'AUPRC': [],
        'RECALL': [],
        'PRECISION': [],
        'F1': [],
        'BRIER': []
    }
    model = MLKD_Model(model_path=student_init_path,
                       class_num=class_num)
    test_list = ['best_F1_model', 'best_BRIER_model', 'best_Loss_model']

    for t_name in test_list:
        print(f'\t>>> {t_name} : ')
        checkpoint = torch.load(os.path.join(ckpt_path, t_name + '.tar'), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict']);
        model.to(device)

        for k in range(bootstrap_K):
            bootstrap_data = inference_data.sample(frac=sample_rate, random_state=SEED + k)
            bootstrap_data = bootstrap_data.reset_index(drop=True)
            bootstrap_dataset = MLKD_Dataset(bootstrap_data,
                                             t_tokenizer,
                                             s_tokenizer,
                                             max_length=max_length,
                                             padding=padding,
                                             return_tensors='pt',
                                             class_num=class_num)
            bootstrap_loader = DataLoader(bootstrap_dataset, batch_size=batch_size,
                                          sampler=RandomSampler(bootstrap_dataset))

            loss, acc, (TH_ACC, AUROC, AUPRC, RECALL, PRECISION, F1, BRIER) = MLKD_evaluate(model,
                                                                                            device,
                                                                                            bootstrap_loader,
                                                                                            label_frequency)
            print(f'\t\tResult : loss:{loss:.4f}, AUPRC:{AUPRC:.4f}, AUROC:{AUROC:.4f}, RECALL:{RECALL:.4f}, PRECISION:{PRECISION:.4f}, BRIER:{BRIER:.4f}')
            METRIC_DICT[t_name]['loss'].append(loss)
            METRIC_DICT[t_name]['acc'].append(acc)
            METRIC_DICT[t_name]['TH_ACC'].append(TH_ACC)
            METRIC_DICT[t_name]['AUROC'].append(AUROC)
            METRIC_DICT[t_name]['AUPRC'].append(AUPRC)
            METRIC_DICT[t_name]['RECALL'].append(RECALL)
            METRIC_DICT[t_name]['PRECISION'].append(PRECISION)
            METRIC_DICT[t_name]['F1'].append(F1)
            METRIC_DICT[t_name]['BRIER'].append(BRIER)

    for key in METRIC_DICT.keys():
        models_metric = METRIC_DICT[key]
        for k in models_metric.keys():
            mean = np.round(np.mean(models_metric[k]), 4)
            std = np.round(np.std(models_metric[k]), 3)

            METRIC_RESULT[k].append(f"{mean}±{std}")
    print(METRIC_RESULT)
    # with open(os.path.join(ckpt_path, 'inference_result.json'),'w') as f:
    #     json.dump(METRIC_DICT, f, indent=4)

    model_result_df = pd.DataFrame(METRIC_RESULT, index=METRIC_DICT.keys())
    model_result_df.to_csv(os.path.join(ckpt_path, 'result.csv'))
    return model_result_df

def main():
    pass
    
if __name__ == '__main__':
    main()