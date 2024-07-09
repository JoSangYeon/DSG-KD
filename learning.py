import os
import sys
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as torch_utils
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, notebook

from sklearn.metrics import (roc_auc_score, 
                             average_precision_score,
                             accuracy_score,  
                             recall_score,
                             precision_score,
                             f1_score,
                             brier_score_loss, # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html
                             confusion_matrix,)

from utils import *

def base_train(scheduler,
               es_obj,
               model,
               device,
               optimizer,
               epochs,
               save_path,
               label_frequency,
               train_loader,
               valid_loader=None):
    model.to(device)
    ce = nn.CrossEntropyLoss()
    for epoch in range(1, epochs+1):
        model.train()
        sum_loss = sum_acc = 0
        valid_loss = valid_acc = 0
        bs = train_loader.batch_size

        # in notebook
        # pabr = notebook.tqdm(enumerate(train_loader), file=sys.stdout)

        # in interpreter
        pbar = tqdm(train_loader, file=sys.stdout)
        for batch_idx, ((input_ids, att_mask, type_ids), target) in enumerate(pbar):
            input_ids, att_mask, type_ids = input_ids.to(device), att_mask.to(device), type_ids.to(device)
            target = target.to(device)
            mb_len = len(target)

            optimizer.zero_grad()
            output = model(input_ids, att_mask, type_ids, labels=target)
            loss, logit = output.loss, output.logits
            acc = calc_acc(logit, target)
            loss.backward()
            torch_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            sum_loss += loss.item()
            sum_acc += acc

            loss = sum_loss / (batch_idx + 1)
            acc = sum_acc / (batch_idx * bs + mb_len)
            pbar.set_postfix(epoch=f'{epoch}/{epochs}', loss='{:.4f}, acc={:.4f}'.format(loss, acc))
        pbar.close()

        train_loss = sum_loss / (batch_idx + 1)
        train_acc = sum_acc / (batch_idx * bs + mb_len)
        wandb.log({'train_loss': train_loss,
                   'train_acc': train_acc},
                  step=epoch)

        if valid_loader is not None:
            valid_loss, valid_acc, (AUROC, AUPRC, TH_ACC, RECALL, PRECISION, F1, BRIER) = base_evaluate(model,
                                                                                                        device,
                                                                                                        valid_loader,
                                                                                                        label_frequency=label_frequency)
            es_obj(metric_factors=[F1, BRIER, valid_loss], model=model)
            wandb.log({'valid_loss': valid_loss,
                       'valid_acc': valid_acc,
                       'valid_thacc': TH_ACC,
                       'valid_auroc': AUROC,
                       'valid_auprc': AUPRC,
                       'valid_recall': RECALL,
                       'valid_precision': PRECISION,
                       'valid_f1_score': F1,
                       'valid_brier': BRIER},
                      step=epoch)
        # Early Stopping
        if es_obj.early_stop:
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch' : epoch,
                'train_loss' : train_loss,
                'train_acc' : train_acc,
                'valid_loss' : valid_loss,
                'valid_acc' : valid_acc,
            }, os.path.join(save_path, f'last_checkpoint_{epoch}.tar'))
            break
        print()

    return es_obj.best_score # [best f1, best brier]


def base_evaluate(model, device, data_loader, label_frequency):
    """
    :param model: your model
    :param device: your device(cuda or cpu)
    :param data_loader: valid or test Datasets
    :param label_frequency: label frequency
    :return: (valid or test) loss and acc
    """
    model.eval()
    sum_loss = sum_acc = 0
    bs = data_loader.batch_size

    predicted = torch.tensor([]).to(device)
    labels = torch.tensor([]).to(device)
    with torch.no_grad():
        # in notebook
        # pabr = notebook.tqdm(enumerate(valid_loader), file=sys.stdout)

        # in interpreter
        pbar = tqdm(data_loader, file=sys.stdout)
        for batch_idx, ((input_ids, att_mask, type_ids), target) in enumerate(pbar):
            input_ids, att_mask, type_ids = input_ids.to(device), att_mask.to(device), type_ids.to(device)
            target = target.to(device)
            mb_len = len(target)

            output = model(input_ids, att_mask, type_ids, labels=target)
            loss, logit = output.loss, output.logits
            acc = calc_acc(logit, target)

            sum_loss += loss.item()
            sum_acc += acc

            loss = sum_loss / (batch_idx + 1)
            acc = sum_acc / (batch_idx * bs + mb_len)
            pbar.set_postfix(loss='{:.4f}, acc={:.4f}'.format(loss, acc))

            predicted = torch.concat([predicted, logit], dim=0)
            labels = torch.concat([labels, target], dim=0)
        pbar.close()

    total_loss = sum_loss / (batch_idx + 1)
    total_acc = sum_acc / (batch_idx * bs + mb_len)

    # predicted_probas = torch.sigmoid(predicted)[:, 1]
    predicted_probas = torch.softmax(predicted, dim=-1)[:, 1]
    predicted_labels = torch.where(predicted_probas >= label_frequency , 1, 0)

    predicted_probas = predicted_probas.detach().cpu().numpy()
    predicted_labels = predicted_labels.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    AUROC = roc_auc_score(labels, predicted_probas)
    AUPRC = average_precision_score(labels, predicted_probas)
    TH_ACC = accuracy_score(labels, predicted_labels)
    RECALL = recall_score(labels, predicted_labels)
    PRECISION = precision_score(labels, predicted_labels)
    F1 = f1_score(labels, predicted_labels)
    BRIER = brier_score_loss(labels, predicted_probas)
    # CM = confusion_matrix(labels, predicted_labels)

    return total_loss, total_acc, (TH_ACC, AUROC, AUPRC, RECALL, PRECISION, F1, BRIER)

def MLKD_train(scheduler,
               es_obj,
               teacher,
               student,
               device,
               criterion,
               optimizer,
               epochs,
               save_path,
               label_frequency,
               train_loader,
               valid_loader=None):
    teacher.to(device); student.to(device)
    for epoch in range(1, epochs+1):
        teacher.eval(); student.train()
        sum_loss = sum_pred_loss = sum_hidn_loss = sum_attn_loss = sum_acc = 0
        valid_loss = valid_acc = 0
        bs = train_loader.batch_size

        pbar = tqdm(train_loader, file=sys.stdout)
        for batch_idx, ((t_input_ids, t_type_ids, t_att_mask),
                        (s_input_ids, s_type_ids, s_att_mask),
                        (teacher_cs_token_align,
                         student_cs_token_align,
                         cs_token_align_len), target) in enumerate(pbar):
            t_input_ids, t_att_mask, t_type_ids = t_input_ids.to(device), t_att_mask.to(device), t_type_ids.to(device)
            s_input_ids, s_att_mask, s_type_ids = s_input_ids.to(device), s_att_mask.to(device), s_type_ids.to(device)
            target = target.to(device)
            mb_len = len(target)

            optimizer.zero_grad()
            with torch.no_grad():
                _, t_hidden_states, t_att_matrices, _ = teacher(t_input_ids,
                                                                t_type_ids,
                                                                t_att_mask)

            ((logit, output),
             (s_hidden_states, s_att_matrices)) = student(s_input_ids,
                                                          s_type_ids,
                                                          s_att_mask)
            hidn_loss, attn_loss, pred_loss = criterion(output, target,
                                                        t_hidden_states, t_att_matrices,
                                                        s_hidden_states, s_att_matrices,
                                                        teacher_cs_token_align,
                                                        student_cs_token_align,
                                                        cs_token_align_len)
            loss = hidn_loss + attn_loss + pred_loss
            acc = calc_acc(output, target)
            loss.backward()
            torch_utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            sum_loss += loss.item()
            sum_pred_loss += pred_loss.item()
            sum_hidn_loss += hidn_loss.item()
            sum_attn_loss += attn_loss.item()
            sum_acc += acc

            loss = sum_loss / (batch_idx + 1)
            pred_loss = sum_pred_loss / (batch_idx + 1)
            hidn_loss = sum_hidn_loss / (batch_idx + 1)
            attn_loss = sum_attn_loss / (batch_idx + 1)
            acc = sum_acc / (batch_idx * bs + mb_len)
            pbar.set_postfix(epoch=f'{epoch}/{epochs}', loss='{:.3f}, pred_loss={:.3f}, hidn_loss={:.3f}, attn_loss={:.3f}, acc={:.3f}'.format(loss,
                                                                                                                                               pred_loss,
                                                                                                                                               hidn_loss,
                                                                                                                                               attn_loss,
                                                                                                                                               acc))
        pbar.close()

        train_loss = sum_loss / (batch_idx + 1)
        train_pred_loss = sum_pred_loss / (batch_idx + 1)
        train_hidn_loss = sum_hidn_loss / (batch_idx + 1)
        train_attn_loss = sum_attn_loss / (batch_idx + 1)
        train_acc = sum_acc / (batch_idx * bs + mb_len)
        wandb.log({'train_loss': train_loss,
                   'train_pred_loss': train_pred_loss,
                   'train_hidn_loss': train_hidn_loss,
                   'train_attn_loss': train_attn_loss,
                   'train_acc': train_acc},
                  step=epoch)

        if valid_loader is not None:
            valid_loss, valid_acc, (AUROC, AUPRC, TH_ACC, RECALL, PRECISION, F1, BRIER) = MLKD_evaluate(student,
                                                                                                        device,
                                                                                                        valid_loader,
                                                                                                        label_frequency)
            es_obj(metric_factors=[F1, BRIER, valid_loss], model=student)
            wandb.log({'valid_loss': valid_loss,
                       'valid_acc': valid_acc,
                       'valid_thacc': TH_ACC,
                       'valid_auroc': AUROC,
                       'valid_auprc': AUPRC,
                       'valid_recall': RECALL,
                       'valid_precision': PRECISION,
                       'valid_f1_score': F1,
                       'valid_brier': BRIER},
                      step=epoch)
        # Early Stopping
        if es_obj.early_stop:
            torch.save({
                'model_state_dict': student.state_dict(),
                'epoch' : epoch,
                'train_loss' : train_loss,
                'train_acc' : train_acc,
                'valid_loss' : valid_loss,
                'valid_acc' : valid_acc,
            }, os.path.join(save_path, f'last_checkpoint_{epoch}.tar'))
            break
        print()

    return es_obj.best_score # [best f1, best brier]

def MLKD_evaluate(model,
                  device,
                  data_loader,
                  label_frequency):
    eval_criterion = nn.CrossEntropyLoss()

    model.eval()
    sum_loss = sum_acc = 0
    bs = data_loader.batch_size

    predicted = torch.tensor([]).to(device)
    labels = torch.tensor([]).to(device)
    with torch.no_grad():
        # in notebook
        # pabr = notebook.tqdm(enumerate(valid_loader), file=sys.stdout)

        # in interpreter
        pbar = tqdm(data_loader, file=sys.stdout)
        for batch_idx, ((_, _, _),
                        (s_input_ids, s_type_ids, s_att_mask),
                        (_, _, _), target) in enumerate(pbar):
            s_input_ids, s_att_mask, s_type_ids = s_input_ids.to(device), s_att_mask.to(device), s_type_ids.to(device)
            target = target.to(device)
            mb_len = len(target)

            ((logit, output), 
             (_, _)) = model(s_input_ids, s_type_ids, s_att_mask)
            loss = eval_criterion(logit, target)
            acc = calc_acc(logit, target)

            sum_loss += loss.item()
            sum_acc += acc

            loss = sum_loss / (batch_idx + 1)
            acc = sum_acc / (batch_idx * bs + mb_len)
            pbar.set_postfix(loss='{:.4f}, acc={:.4f}'.format(loss, acc))

            predicted = torch.concat([predicted, output], dim=0)
            labels = torch.concat([labels, target], dim=0)
        pbar.close()

    total_loss = sum_loss / (batch_idx + 1)
    total_acc = sum_acc / (batch_idx * bs + mb_len)

    # predicted_probas = torch.sigmoid(predicted)[:, 1]
    predicted_probas = torch.softmax(predicted, dim=-1)[:, 1]
    predicted_labels = torch.where(predicted_probas >= label_frequency, 1, 0)

    predicted_probas = predicted_probas.detach().cpu().numpy()
    predicted_labels = predicted_labels.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    AUROC = roc_auc_score(labels, predicted_probas)
    AUPRC = average_precision_score(labels, predicted_probas)
    TH_ACC = accuracy_score(labels, predicted_labels)
    RECALL = recall_score(labels, predicted_labels)
    PRECISION = precision_score(labels, predicted_labels)
    F1 = f1_score(labels, predicted_labels)
    BRIER = brier_score_loss(labels, predicted_probas)
    # CM = confusion_matrix(labels, predicted_labels)

    return total_loss, total_acc, (TH_ACC, AUROC, AUPRC, RECALL, PRECISION, F1, BRIER)

def main():
    pass


if __name__ == "__main__":
    main()
