# https://github.com/HideOnHouse/TorchBase

import os
import wandb
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, random_split

# from transformers import AdamW
from transformers import BertTokenizer
from transformers import AutoTokenizer
# from transformers import AdamW
from transformers import get_cosine_schedule_with_warmup
from transformers import get_linear_schedule_with_warmup

from dataset import *
from learning import *
from inference import *
from utils import *

import warnings
warnings.filterwarnings(action='ignore')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--random_seed", type=int, default=17)

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--method", type=str, default='base') # or mlkd or ensemble
    parser.add_argument("--student", type=str, default='kobert') # or kmbert
    parser.add_argument("--teacher", type=str, default='mbert')
    parser.add_argument("--model", type=str, default='kobert') # or kmbert, bert, mbert, cbert
    parser.add_argument("--vocab_type", type=str, default='S') # if model==kmbert
    parser.add_argument("--is_pt", type=str2bool, default=False)

    parser.add_argument("--bert_lr", type=float, default=1e-5)
    parser.add_argument("--classifier_lr", type=float, default=1e-2)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.2)

    parser.add_argument("--max_seq_length", type=int, default=384)
    parser.add_argument("--padding", type=str, default='max_length')
    parser.add_argument("--cs_max_len", type=int, default=32)
    parser.add_argument("--class_num", type=int, default=2)

    args = parser.parse_args()

    # args
    SEED = args.random_seed; set_SEED(SEED)

    epochs = args.epochs
    batch_size = args.batch_size

    method = args.method
    student = args.student
    teacher = args.teacher
    model = args.model

    vocab_type = args.vocab_type
    is_pt = args.is_pt

    bert_lr = args.bert_lr
    classifier_lr = args.classifier_lr
    alpha = args.alpha
    beta = args.beta

    max_length = args.max_seq_length
    padding = args.padding
    cs_max_len = args.cs_max_len
    class_num = args.class_num

    # GPU Device
    device = set_device(device_num=0)

    # loading model
    if method == 'base':
        if model == 'kobert':
            model_init_path = 'kykim/bert-kor-base'
            vocab_path = 'kykim/bert-kor-base'
            model_name = "Base_Ko_BERT"
        elif model == 'kmbert':
            if vocab_type == 'B' and not is_pt:
                model_init_path = os.path.join('..', 'kmbert_B')
                vocab_path = os.path.join('..', 'kmbert_B')
                model_name = f"Base_KM_BERT_{vocab_type}"
            elif vocab_type == 'S' and not is_pt:
                model_init_path = os.path.join('..', 'kmbert_S')
                vocab_path = os.path.join('..', 'kmbert_S')
                model_name = f"Base_KM_BERT_{vocab_type}"
            elif vocab_type == 'B' and is_pt:
                model_init_path = os.path.join('..', 'kmbert_B_new', 'checkpoint-27000')
                vocab_path = os.path.join('..', 'kmbert_B_new')
                model_name = f"Base_MLM_KM_BERT_{vocab_type}"
            elif vocab_type == 'S' and is_pt:
                model_init_path = os.path.join('..', 'kmbert_S_new', 'checkpoint-48000')
                vocab_path = os.path.join('..', 'kmbert_S_new')
                model_name = f"Base_MLM_KM_BERT_{vocab_type}"
            else:
                print('Wrong vocab_type or is_pt')
                return
        elif model == 'bert':
            model_init_path = 'bert-base-uncased'
            vocab_path = 'bert-base-uncased'
            model_name = f"Base_BERT"
        elif model == 'mbert':
            model_init_path = 'bert-base-multilingual-uncased'
            vocab_path = 'bert-base-multilingual-uncased'
            model_name = f"Base_M_BERT"
        elif model == 'cbert':
            model_init_path = 'emilyalsentzer/Bio_ClinicalBERT'
            vocab_path = 'emilyalsentzer/Bio_ClinicalBERT'
            model_name = "Base_Clincal_BERT"
        else:
            print('Wrong model_name : only [kobert, kmbert, bert, mbert, cbert]')
            return
    elif method == 'mlkd' or method == 'ensemble':
        if student == 'kobert':
            student_model_init_path = 'kykim/bert-kor-base'
            student_vocab_path = 'kykim/bert-kor-base'
            stu = 'Ko_BERT'
        elif student == 'kmbert':
            if vocab_type == 'B' and not is_pt:
                student_model_init_path = os.path.join('..', 'kmbert_B')
                student_vocab_path = os.path.join('..', 'kmbert_B')
                stu = 'KM_BERT_B'
            elif vocab_type == 'S' and not is_pt:
                student_model_init_path = os.path.join('..', 'kmbert_S')
                student_vocab_path = os.path.join('..', 'kmbert_S')
                stu = 'KM_BERT_S'
            elif vocab_type == 'B' and is_pt:
                student_model_init_path = os.path.join('..', 'kmbert_B_new', 'checkpoint-27000')
                student_vocab_path = os.path.join('..', 'kmbert_B_new')
                stu = 'MLM_KM_BERT_B'
            elif vocab_type == 'S' and is_pt:
                student_model_init_path = os.path.join('..', 'kmbert_S_new', 'checkpoint-48000')
                student_vocab_path = os.path.join('..', 'kmbert_S_new')
                stu = 'MLM_KM_BERT_S'
            else:
                print("Wrong student\'s vocab_type or is_pt")
                return
        elif student == 'mbert':
            student_model_init_path = os.path.join('..', 'mbert')
            student_vocab_path = os.path.join('..', 'mbert')
            stu = 'M_BERT'
        else:
            print('Wrong student model_name : only [kobert, kmbert, mbert]')

        if teacher == 'bert':
            teacher_model_init_path = 'bert-base-uncased'
            teacher_vocab_path = 'bert-base-uncased'
            tea = "BERT"
        elif teacher == 'mbert':
            teacher_model_init_path = 'bert-base-multilingual-uncased'
            teacher_vocab_path = 'bert-base-multilingual-uncased'
            tea = 'M_BERT'
        elif teacher == 'cbert':
            teacher_model_init_path = 'emilyalsentzer/Bio_ClinicalBERT'
            teacher_vocab_path = 'emilyalsentzer/Bio_ClinicalBERT'
            tea = 'Clinical_BERT'
        elif teacher == 'roberta':
            teacher_model_init_path = 'roberta-base'
            teacher_vocab_path = 'roberta-base'
            tea = 'RoBERTa'
        elif teacher == 'bio_roberta':
            teacher_model_init_path = 'allenai/biomed_roberta_base'
            teacher_vocab_path = 'allenai/biomed_roberta_base'
            tea = 'Bio_RoBERTa'
        elif teacher == 'bio_mbert':
            teacher_model_init_path = 'StivenLancheros/mBERT-base-Biomedical-NER'
            teacher_vocab_path = 'StivenLancheros/mBERT-base-Biomedical-NER'
            tea = 'Bio_M_BERT'
        else:
            print('Wrong teacher model_name : only [bert, mbert, cbert, roberta]')
            return

        if method == 'mlkd':
            model_name = f"MLKD_{stu}_{tea}"
        else: # method == 'ensemble'
            model_name = f"MLKD_ensemble_{stu}_{tea}"
    else:
        print('Wrong method : only [base, mlkd, ensemble]')
        return
    save_path = set_save_path(model_name, epochs, batch_size)

    # Define project
    project_name = 'Project-MLKD_TEST'
    wandb.init(project=project_name)
    wandb.run.name = model_name

    # load data
    train_data, valid_data, test_data, label_frequency = load_data(
        train_x_path=os.path.join("..", "data(v5)", "1", "X_train.csv"),
        train_y_path=os.path.join("..", "data(v5)", "1", "y_train.csv"),
        valid_x_path=os.path.join("..", "data(v5)", "1", "X_val.csv"),
        valid_y_path=os.path.join("..", "data(v5)", "1", "y_val.csv"),
        test_x_path=os.path.join("..", "data(v5)", "1", "X_test.csv"),
        test_y_path=os.path.join("..", "data(v5)", "1", "y_test.csv"), )

    if method == 'base':
        config = {
            'bert_lr': bert_lr,
            'classifier_lr': classifier_lr,
            'batch_size': batch_size,
            'Label_frequency': label_frequency,
            'max_length': max_length,
            'epochs': epochs,
        }; wandb.config.update(config)
        
        ########################### Just Code Checking ###########################
        check_rate = 0.75
        train_data = train_data.sample(frac=check_rate, random_state=SEED).reset_index(drop=True)
        valid_data = valid_data.sample(frac=check_rate, random_state=SEED).reset_index(drop=True)
        test_data = test_data.sample(frac=check_rate, random_state=SEED).reset_index(drop=True)
        
        label_frequency = train_data['LABEL'].sum() / len(train_data)
        print('THRESHOLD : {}'.format(label_frequency))
        # check_sz = int(len(train_dataset) * check_rate)
        # non_check_sz = len(train_dataset) - check_sz
        # train_dataset, _ = random_split(train_dataset, [check_sz, non_check_sz])

        # # label_frequency = train_dataset.dataset.y_data.sum() / len(train_dataset)
        # # print('label_frequency : {}'.format(label_frequency))

        # check_sz = int(len(valid_dataset) * check_rate)
        # non_check_sz = len(valid_dataset) - check_sz
        # valid_dataset, _ = random_split(valid_dataset, [check_sz, non_check_sz])

        # check_sz = int(len(test_dataset) * check_rate)
        # non_check_sz = len(test_dataset) - check_sz
        # test_dataset, _ = random_split(test_dataset, [check_sz, non_check_sz])
        ##########################################################################

        tokenizer = BertTokenizer.from_pretrained(vocab_path)

        train_dataset = Base_Dataset(train_data,
                                     tokenizer,
                                     max_length=max_length,
                                     padding=padding,
                                     return_tensors='pt',
                                    class_num=class_num)
        valid_dataset = Base_Dataset(valid_data,
                                     tokenizer,
                                     max_length=max_length,
                                     padding=padding,
                                     return_tensors='pt',
                                    class_num=class_num)
        test_dataset = Base_Dataset(test_data,
                                    tokenizer,
                                    max_length=max_length,
                                    padding=padding,
                                    return_tensors='pt',
                                    class_num=class_num)

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  sampler=RandomSampler(train_dataset))
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                                  sampler=RandomSampler(valid_dataset))
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 sampler=RandomSampler(test_dataset))

        # modeling #
        from transformers import BertForSequenceClassification
        model = BertForSequenceClassification.from_pretrained(model_init_path, num_labels=class_num)
        model.to(device)
        optimizer = optim.AdamW([{'params': model.bert.parameters(), 'lr': bert_lr},
                                 {'params': model.classifier.parameters(), 'lr': classifier_lr}],
                                eps=1e-8)

        # optimizer = optim.AdamW(model.parameters(), lr=5e-5, eps=1e-8)
        iter_len = len(train_loader)
        num_training_steps = iter_len * epochs
        num_warmup_steps = int(0.15 * num_training_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps)
        es_obj = EarlyStopping(patience=5,
                               verbose=False,
                               delta=0,
                               num_factors=3,  # [f1, brier, Loss]
                               save_path=save_path)

        # train
        print("============================= Train =============================")
        best_f1, best_brier, best_loss = base_train(scheduler, es_obj, model, device, optimizer, epochs, save_path,
                                         label_frequency, train_loader, valid_loader)
        print('Train Best F1_Score : {:.5f}'.format(best_f1))
        print('Train Best Brier_SCORE : {:.5f}'.format(best_brier))
        print('Train Best Loss : {:.5f}'.format(best_loss))

        # Test
        print("============================= Test =============================")
        test_loss, test_acc, (TH_ACC, AUROC, AUPRC, RECALL, PRECISION, F1, BRIER) = base_evaluate(model,
                                                                                                  device,
                                                                                                  test_loader,
                                                                                                  label_frequency)
        print("test loss : {:.4f}".format(test_loss))
        print("test acc : {:.4f}".format(test_acc))
        print("test acc(th) : {:4f}".format(TH_ACC))
        print("test AUROC : {:.4f}".format(AUROC))
        print("test AUPRC : {:.4f}".format(AUPRC))
        print("test Recall : {:4f}".format(RECALL))
        print("test Precision : {:.4f}".format(PRECISION))
        print("test F1_score : {:.4f}".format(F1))
        print("test Brier : {:4f}".format(BRIER))

        # Inference Best Model
        metric_result = base_inference(model_init_path, save_path, device, test_data, tokenizer,
                                       label_frequency, sample_rate=0.75, bootstrap_K=10,
                                       max_length=max_length, padding=padding, class_num=class_num,
                                       batch_size=batch_size, SEED=SEED)

        print(metric_result)
        result_table = wandb.Table(dataframe=metric_result)
        wandb.log({'METRIC_RESULT': result_table})

    elif method == 'mlkd':
        config = {
            'bert_lr': bert_lr,
            'classifier_lr': classifier_lr,
            'alpha': alpha,
            'beta': beta,
            'batch_size': batch_size,
            'Label_frequency': label_frequency,
            'max_length': max_length,
            'epochs': epochs,
        }; wandb.config.update(config)
        
        ########################### Just Code Checking ###########################
        check_rate = 0.75
        train_data = train_data.sample(frac=check_rate, random_state=SEED).reset_index(drop=True)
        valid_data = valid_data.sample(frac=check_rate, random_state=SEED).reset_index(drop=True)
        test_data = test_data.sample(frac=check_rate, random_state=SEED).reset_index(drop=True)
        
        label_frequency = train_data['LABEL'].sum() / len(train_data)
        print('THRESHOLD : {}'.format(label_frequency))
        # check_sz = int(len(train_dataset) * check_rate)
        # non_check_sz = len(train_dataset) - check_sz
        # train_dataset, _ = random_split(train_dataset, [check_sz, non_check_sz])

        # # label_frequency = train_dataset.dataset.y_data.sum() / len(train_dataset)
        # # print('label_frequency : {}'.format(label_frequency))

        # check_sz = int(len(valid_dataset) * check_rate)
        # non_check_sz = len(valid_dataset) - check_sz
        # valid_dataset, _ = random_split(valid_dataset, [check_sz, non_check_sz])

        # check_sz = int(len(test_dataset) * check_rate)
        # non_check_sz = len(test_dataset) - check_sz
        # test_dataset, _ = random_split(test_dataset, [check_sz, non_check_sz])
        ##########################################################################

        t_tokenizer = AutoTokenizer.from_pretrained(teacher_vocab_path)
        s_tokenizer = AutoTokenizer.from_pretrained(student_vocab_path)

        train_dataset = MLKD_Dataset(train_data,
                                     t_tokenizer,
                                     s_tokenizer,
                                     max_length=max_length,
                                     padding=padding,
                                     return_tensors='pt',
                                     cs_max_len=cs_max_len,
                                     class_num=class_num)
        valid_dataset = MLKD_Dataset(valid_data,
                                     t_tokenizer,
                                     s_tokenizer,
                                     max_length=max_length,
                                     padding=padding,
                                     return_tensors='pt',
                                     cs_max_len=cs_max_len,
                                     class_num=class_num)
        test_dataset = MLKD_Dataset(test_data,
                                    t_tokenizer,
                                    s_tokenizer,
                                    max_length=max_length,
                                    padding=padding,
                                    return_tensors='pt',
                                    cs_max_len=cs_max_len,
                                    class_num=class_num)

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  sampler=RandomSampler(train_dataset))
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                                  sampler=RandomSampler(valid_dataset))
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 sampler=RandomSampler(test_dataset))

        # modeling #
        from model import MLKD_Model, MLKD_Model_for_Classification
        teacher = MLKD_Model(model_path=teacher_model_init_path)
        student = MLKD_Model_for_Classification(model_path=student_model_init_path,
                                                class_num=class_num)

        # param_optimizer = list(student.named_parameters())
        # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        # ]
        # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=1e-3)
        optimizer = optim.AdamW([{'params': student.bert.parameters(), 'lr': bert_lr},
                                 {'params': student.classifier.parameters(), 'lr': classifier_lr}],
                                eps=1e-8)
        criterion = MLKD_Loss(device=device, alpha=alpha, beta=beta)
        iter_len = len(train_loader)
        num_training_steps = iter_len * epochs
        num_warmup_steps = int(0.25 * num_training_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps)
        es_obj = EarlyStopping(patience=5,
                               verbose=False,
                               delta=0,
                               num_factors=3,  # [f1, brier, Loss]
                               save_path=save_path)

        # train
        print("============================= Train =============================")
        best_f1, best_brier, best_loss = MLKD_train(scheduler, es_obj, teacher, student, device, criterion, optimizer,
                                         epochs, save_path, label_frequency, train_loader, valid_loader)
        print('Train Best F1_Score : {:.5f}'.format(best_f1))
        print('Train Best Brier_SCORE : {:.5f}'.format(best_brier))
        print('Train Best Loss : {:.5f}'.format(best_loss))

        # Test
        print("============================= Test =============================")
        test_loss, test_acc, (TH_ACC, AUROC, AUPRC, RECALL, PRECISION, F1, BRIER) = MLKD_evaluate(student,
                                                                                                  device,
                                                                                                  test_loader,
                                                                                                  label_frequency)
        print("test loss : {:.4f}".format(test_loss))
        print("test acc : {:.4f}".format(test_acc))
        print("test acc(th) : {:4f}".format(TH_ACC))
        print("test AUROC : {:.4f}".format(AUROC))
        print("test AUPRC : {:.4f}".format(AUPRC))
        print("test Recall : {:4f}".format(RECALL))
        print("test Precision : {:.4f}".format(PRECISION))
        print("test F1_score : {:.4f}".format(F1))
        print("test Brier : {:4f}".format(BRIER))

        # Inference Best Model
        metric_result = MLKD_inference(student_model_init_path, t_tokenizer, s_tokenizer,
                                       save_path, device, test_data, label_frequency,
                                       sample_rate=0.75, bootstrap_K=10, max_length=max_length, padding=padding,
                                       class_num=class_num, batch_size=batch_size, SEED=SEED)

        print(metric_result)
        result_table = wandb.Table(dataframe=metric_result)
        wandb.log({'METRIC_RESULT': result_table})
    else:
        print('Wrong method : only [base, mlkd]')
        return

if __name__ == '__main__':
    main()
    # CUDA_VISIBLE_DEVICES=1 python main.py --random_seed 17 --epochs 15 --batch_size 32 --method mlkd --student kobert --teacher roberta --bert_lr 1e-4 --classifier_lr 1e-3