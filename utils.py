import numpy as np
import torch
import torch.nn as nn

import os
import pandas as pd
import matplotlib.pyplot as plt

def load_data(train_x_path = os.path.join("..", "data(v5)", "1", "X_train.csv"),
              train_y_path = os.path.join("..", "data(v5)", "1", "y_train.csv"),
              valid_x_path = os.path.join("..", "data(v5)", "1", "X_val.csv"),
              valid_y_path = os.path.join("..", "data(v5)", "1", "y_val.csv"),
              test_x_path = os.path.join("..", "data(v5)", "1", "X_test.csv"),
              test_y_path = os.path.join("..", "data(v5)", "1", "y_test.csv"),):

    train_x = pd.read_csv(train_x_path, low_memory=False)  # index_col = 0?
    train_y = pd.read_csv(train_y_path, low_memory=False)
    valid_x = pd.read_csv(valid_x_path, low_memory=False)
    valid_y = pd.read_csv(valid_y_path, low_memory=False)
    test_x = pd.read_csv(test_x_path, low_memory=False)
    test_y = pd.read_csv(test_y_path, low_memory=False)

    x_train = train_x.filter(['ER_DHX'])
    y_train = train_y.filter(['LABEL'])
    x_valid = valid_x.filter(['ER_DHX'])
    y_valid = valid_y.filter(['LABEL'])
    x_test = test_x.filter(['ER_DHX'])
    y_test = test_y.filter(['LABEL'])

    train_data = x_train.assign(LABEL=y_train)
    valid_data = x_valid.assign(LABEL=y_valid)
    test_data = x_test.assign(LABEL=y_test)

    train_data = train_data.dropna(axis=0)
    train_data = train_data.reset_index(drop=True)
    valid_data = valid_data.dropna(axis=0)
    valid_data = valid_data.reset_index(drop=True)
    test_data = test_data.dropna(axis=0)
    test_data = test_data.reset_index(drop=True)

    threshold = train_data['LABEL'].sum() / len(train_data)
    print('THRESHOLD : {}'.format(threshold))

    print(train_x.shape, train_y.shape)
    print(valid_x.shape, valid_y.shape)
    print(test_x.shape, test_y.shape)

    return train_data, valid_data, test_data, threshold

def set_SEED(SEED=17):
    # random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def calc_acc(output, label):
    o_val, o_idx = torch.max(output, dim=-1)
    # l_val, l_idx = torch.max(label, dim=-1)
    return (o_idx == label).sum().item()

def draw_history(history, save_path=None):
    train_loss = history["train_loss"]
    train_acc = history["train_acc"]
    valid_loss = history["valid_loss"]
    valid_acc = history["valid_acc"]

    plt.subplot(2,1,1)
    plt.ylabel('Loss')
    plt.plot(train_loss, label="train")
    plt.plot(valid_loss, label="valid")
    plt.legend()

    plt.subplot(2,1,2)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.plot(train_acc, label="train")
    plt.plot(valid_acc, label="valid")
    plt.legend()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(os.path.join(save_path, 'train_plot.png'), dpi=300)

def set_device(device_num=0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        device += ':{}'.format(device_num)
    return device

def set_save_path(model_name, epochs, batch_size):
    directory = os.path.join('models', f'{model_name}_e{epochs}_bs{batch_size}')
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)
    return directory

class MLKD_Loss(nn.Module):
    def __init__(self, device, alpha=0.1, beta=0.1):
        super(MLKD_Loss, self).__init__()
        
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

        self.avg_pool_h = nn.AdaptiveAvgPool2d((1, None))
        self.avg_pool_v = nn.AdaptiveAvgPool2d((None, 1))

        self.device = device
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, voted_logit, target,
                t_hidden_states, t_att_matrices,
                s_hidden_states, s_att_matrices,
                teacher_cs_token_align, # (batch, max_cs_len, 2)
                student_cs_token_align, # (batch, max_cs_len, 2)
                cs_token_align_len): # (batch,)
        # hidden_states : each encoders output : size(layer_num+1, batch, seq_len, 768) # +1 is Embedding Layers
        # att_mterices : each multi-head attention's Q*K^T : size(layer_num, batch, head_num, seq_len, seq_len)

        hidden_size = t_hidden_states[0].size(-1)
        seq_len = t_att_matrices[0].size(-1)

        t_att_matrices = torch.where(t_att_matrices <= -1e2,
                                     torch.zeros_like(t_att_matrices).to(self.device),
                                     t_att_matrices)
        s_att_matrices = torch.where(s_att_matrices <= -1e2,
                                     torch.zeros_like(s_att_matrices).to(self.device),
                                     s_att_matrices)

        teacher_hidn_cs_embeddings = []
        teacher_attn_cs_embeddings = []

        student_hidn_cs_embeddings = []
        student_attn_cs_embeddings = []

        for batch_idx, (t_cs, s_cs, cs_len) in enumerate(zip(teacher_cs_token_align,
                                                             student_cs_token_align,
                                                             cs_token_align_len)):
            if cs_len == 0:
                continue
            for cs_idx, ((t_cs_s, t_cs_e),
                         (s_cs_s, s_cs_e)) in enumerate(zip(t_cs, s_cs)):
                if cs_idx == cs_len:
                    break

                # CS Hidden state #
                # if 0 in t_hidden_states[:, batch_idx, t_s:t_e, :].shape:
                #     print(t_s, t_e, s_s, s_e)
                #     print(t_hidden_states[:, batch_idx, t_s:t_e, :].shape)
                t_hidn_cs_embedding = self.avg_pool_h(t_hidden_states[:, batch_idx, t_cs_s:t_cs_e, :]).view(-1, hidden_size)
                s_hidn_cs_embedding = self.avg_pool_h(s_hidden_states[:, batch_idx, s_cs_s:s_cs_e, :]).view(-1, hidden_size)

                teacher_hidn_cs_embeddings.append(t_hidn_cs_embedding)
                student_hidn_cs_embeddings.append(s_hidn_cs_embedding)

                # # CS Attention Matrices #
                t_attn_cs_h_embedding = self.avg_pool_h(t_att_matrices[:, batch_idx, :, t_cs_s:t_cs_e, :]).view(-1, seq_len)
                # t_attn_cs_v_embedding = self.avg_pool_v(t_att_matrices[:, batch_idx, :, :, t_cs_s:t_cs_e]).view(-1, seq_len)

                s_attn_cs_h_embedding = self.avg_pool_h(s_att_matrices[:, batch_idx, :, s_cs_s:s_cs_e, :]).view(-1, seq_len)
                # s_attn_cs_v_embedding = self.avg_pool_v(s_att_matrices[:, batch_idx, :, :, s_cs_s:s_cs_e]).view(-1, seq_len)

                teacher_attn_cs_embeddings.append(t_attn_cs_h_embedding)
                # teacher_attn_cs_embeddings.append(t_attn_cs_v_embedding)

                student_attn_cs_embeddings.append(s_attn_cs_h_embedding)
                # student_attn_cs_embeddings.append(s_attn_cs_v_embedding)

        teacher_hidn_cs_embeddings = torch.cat(teacher_hidn_cs_embeddings, dim=0)
        teacher_attn_cs_embeddings = torch.cat(teacher_attn_cs_embeddings, dim=0)

        student_hidn_cs_embeddings = torch.cat(student_hidn_cs_embeddings, dim=0)
        student_attn_cs_embeddings = torch.cat(student_attn_cs_embeddings, dim=0)

        assert teacher_hidn_cs_embeddings.shape == student_hidn_cs_embeddings.shape
        assert teacher_attn_cs_embeddings.shape == student_attn_cs_embeddings.shape

        ### hidn_loss ###
        if teacher_hidn_cs_embeddings.size(0) > 0:
            hidn_loss = self.alpha * self.mse(teacher_hidn_cs_embeddings,
                                              student_hidn_cs_embeddings)
        else:
            hidn_loss = torch.tensor(0).float()

        ### attn_loss ###
        if teacher_attn_cs_embeddings.size(0) > 0:
            attn_loss = self.beta * self.mse(teacher_attn_cs_embeddings,
                                             student_attn_cs_embeddings)
        else:
            attn_loss = torch.tensor(0).float()

        ### pred_loss ###
        pred_loss = self.ce(voted_logit, target)
        
        return hidn_loss, attn_loss, pred_loss


class EarlyStopping:
    '''
    patience: epoch을 도는동안 성능 개선이 없을때, 돌 epoch의 횟수 
              default = 5로 설정
    verbose : EarlyStopping의 진행 상황에 대한 정보를 출력할지 여부
              verbose가 True로 설정되면, 검증 손실이 개선되지 않은 에포크 수를 출력하고, 모델이 저장되는 경우에도 출력
              default = True
    delta : delta  (float): 개선시 인정되는 최소 변화 수치. default: 0.0
    '''

    def __init__(self, patience=5,
                 verbose=False,
                 delta=0,
                 num_factors=2,
                 save_path='models/'):
        self.patience = patience
        self.verbose = verbose

        self.counter = [0 for i in range(num_factors)]  # [f1, brier]
        self.best_score = [None for i in range(num_factors)]  # [f1, brier]
        self.ckpt_mark = [0 for i in range(num_factors)]
        self.early_stop_flag = [False for i in range(num_factors)]

        self.early_stop = False
        self.delta = delta
        self.num_factors = num_factors
        self.save_path = save_path

        self.factors = {0: 'F1', 1: 'BRIER', 2: 'Loss'}

    def __call__(self, metric_factors, model):
        """
        :param metric_factors: [Loss_value, AUPRC_value, AUROC_value] : type=list
        :param model: your model
        """

        for i, val in enumerate(metric_factors):
            if self.best_score[i] is None:
                self.save_checkpoint(i, val, model)
                self.best_score[i] = val
            else:
                if self.factors[i] == 'BRIER' or self.factors[i] == 'Loss':  # Loss는 계산을 다르게함
                    score = -val
                    best_score = -self.best_score[i]
                else:
                    score = val
                    best_score = self.best_score[i]

                if score < best_score + self.delta:  # Best 값을 갱신하지 못했을 경우
                    self.counter[i] += 1
                    if self.verbose:
                        print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                    if self.counter[i] >= self.patience:
                        self.early_stop_flag[i] = True
                else:  # Best값을 갱신한 경우
                    self.save_checkpoint(i, val, model)
                    self.best_score[i] = val
                    self.counter[i] = 0
        self.early_stop = all(self.early_stop_flag)

    def save_checkpoint(self, factor_num, metric_value, model):
        ckpt_savepath = os.path.join(self.save_path, f'best_{self.factors[factor_num]}_model.tar')
        if self.verbose:
            print(f'Validation loss decreased ({self.best_score[factor_num]} --> {metric_value}). Saving model ...')

        torch.save({
            'model_state_dict': model.state_dict(),
            'metric_value': metric_value
        }, ckpt_savepath)