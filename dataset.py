import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class Base_Dataset(Dataset):
    def __init__(self,
                 data,
                 tokenizer,
                 max_length=512,
                 padding='max_length',
                 return_tensors='pt',
                 class_num=2):
        super(Base_Dataset, self).__init__()
        self.data = data
        self.x_data = data.ER_DHX
        self.y_data = data.LABEL #label

        self.tokenizer = tokenizer

        self.max_length = max_length
        self.padding = padding
        self.return_tensors = return_tensors

        self.class_num = class_num
        self.vocab_size = len(self.tokenizer)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        ## sentence ##
        x = self.x_data.iloc[idx]
        tokenizer_output = self.tokenizer(x, max_length=self.max_length, padding=self.padding,
                                          return_tensors=self.return_tensors, truncation=True)

        input_ids = tokenizer_output['input_ids'][0]
        att_mask = tokenizer_output['attention_mask'][0]
        type_ids = tokenizer_output['token_type_ids'][0]

        ## label ##
        y = torch.tensor(self.y_data[idx]).long()
        # y = F.one_hot(y, num_classes=self.class_num).float()

        return (input_ids, att_mask, type_ids), y

class MLKD_Dataset(Dataset):
    def __init__(self,
                 data,
                 t_tokenizer,
                 s_tokenizer,
                 max_length=512,
                 padding='max_length',
                 return_tensors='pt',
                 cs_max_len = 32,
                 class_num=2):
        super(MLKD_Dataset, self).__init__()
        self.data = data
        self.x_data = data.ER_DHX
        self.y_data = data.LABEL #label

        self.t_tokenizer = t_tokenizer
        self.s_tokenizer = s_tokenizer

        self.max_length = max_length
        self.padding = padding
        self.return_tensors = return_tensors

        self.cs_max_len = cs_max_len
        self.class_num = class_num
        self.t_vocab_size = len(self.t_tokenizer)
        self.s_vocab_size = len(self.s_tokenizer)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ## sentence ##
        x = self.x_data[idx]

        teacher_cs_token_align, student_cs_token_align, cs_token_align_len = self._get_cs_token_info(x)

        teacher_tokenizer_output = self.t_tokenizer(x, max_length=self.max_length, padding=self.padding,
                                                    return_tensors=self.return_tensors, truncation=True,
                                                    return_token_type_ids=True, return_attention_mask=True)

        student_tokenizer_output = self.s_tokenizer(x, max_length=self.max_length, padding=self.padding,
                                                    return_tensors=self.return_tensors, truncation=True,
                                                    return_token_type_ids=True, return_attention_mask=True)

        t_input_ids = teacher_tokenizer_output['input_ids'][0]
        t_att_mask = teacher_tokenizer_output['attention_mask'][0]
        t_type_ids = teacher_tokenizer_output['token_type_ids'][0]

        s_input_ids = student_tokenizer_output['input_ids'][0]
        s_att_mask = student_tokenizer_output['attention_mask'][0]
        s_type_ids = student_tokenizer_output['token_type_ids'][0]

        ## label ##
        y = torch.tensor(self.y_data[idx]).long()
        # y = F.one_hot(y, num_classes=self.class_num).float()

        # a = [t_input_ids, t_type_ids, t_att_mask, s_input_ids, s_type_ids, s_att_mask, teacher_cs_token_align, student_cs_token_align, cs_token_align_len, y]
        # for x in a:
        #     print(x.shape)

        return (t_input_ids, t_type_ids, t_att_mask), (s_input_ids, s_type_ids, s_att_mask), (teacher_cs_token_align, student_cs_token_align, cs_token_align_len), y

    def _get_cs_token_info(self, text):
        word_origin = [word for word in text.split(' ')]

        teacher_cs_token_align = []
        student_cs_token_align = []
        # cs_token_align_len = [] # [teacher, student]
        t_s_idx, t_e_idx = 1, 1
        s_s_idx, s_e_idx = 1, 1
        t_cs_cnt, s_cs_cnt = 0, 0
        for word in word_origin:
            t_tokens = self.t_tokenizer.tokenize(word)
            # t_tokens_ids = self.t_tokenizer.convert_ids_to_tokens(t_tokens)
            s_tokens = self.s_tokenizer.tokenize(word)
            # s_tokens_ids = self.s_tokenizer.convert_ids_to_tokens(s_tokens)

            t_s_idx = t_e_idx
            t_e_idx += len(t_tokens)
            s_s_idx = s_e_idx
            s_e_idx += len(s_tokens)
            # TC 1 : 510, 512,

            if t_s_idx >= self.max_length-1 or s_s_idx >= self.max_length-1:
                break

            if word.encode().isalpha():
                if t_e_idx >= self.max_length:
                    teacher_cs_token_align.append([t_s_idx, self.max_length-1]) # -1 is [SEP]
                else:
                    teacher_cs_token_align.append([t_s_idx, t_e_idx])

                if s_e_idx >= self.max_length:
                    student_cs_token_align.append([s_s_idx, self.max_length-1]) # -1 is [SEP]
                else:
                    student_cs_token_align.append([s_s_idx, s_e_idx])
                t_cs_cnt += 1; s_cs_cnt += 1

            if t_cs_cnt > self.cs_max_len or s_cs_cnt > self.cs_max_len:
                teacher_cs_token_align = teacher_cs_token_align[:self.cs_max_len]
                student_cs_token_align = student_cs_token_align[:self.cs_max_len]
                break

        while len(teacher_cs_token_align) < self.cs_max_len:
            teacher_cs_token_align.append([0, 0])
        while len(student_cs_token_align) < self.cs_max_len:
            student_cs_token_align.append([0, 0])

        assert t_cs_cnt == s_cs_cnt
        cs_token_align_len = t_cs_cnt  # teacher == student

        assert len(teacher_cs_token_align) == self.cs_max_len
        assert len(student_cs_token_align) == self.cs_max_len

        return torch.tensor(teacher_cs_token_align).long(), torch.tensor(student_cs_token_align).long(), torch.tensor(cs_token_align_len).long()

class KD_DST(Dataset):
    def __init__(self, 
                 x_data, 
                 y_data,
                 t_tokenizer,                 
                 s_tokenizer,
                 max_length=512,
                 padding='max_length',
                 class_num=2,):
        super(KD_DST, self).__init__()
        self.x_data = x_data.ER_DHX
        self.y_data = y_data.LABEL
        self.class_num = class_num

        self.t_tokenizer = t_tokenizer
        self.s_tokenizer = s_tokenizer

        self.max_length = max_length
        self.padding = padding
        self.return_tensors = 'pt'
        self.return_token_type_ids = False
        self.return_attention_mask = True

    def get_eng_tokens_mask(self, text):
        ignore_list = ['[CLS]', '[SEP]', '[PAD]', '[MASK]', '[UNK]']

        teacher_token_ids = self.t_tokenizer.encode(text, truncation=True, max_length=self.max_length)
        student_token_ids = self.s_tokenizer.encode(text, truncation=True, max_length=self.max_length)

        teacher_token = self.t_tokenizer.convert_ids_to_tokens(teacher_token_ids)
        student_token = self.s_tokenizer.convert_ids_to_tokens(student_token_ids)

        eng_mask = torch.zeros(self.max_length) # (seq_len,)
        for idx, (t, s) in enumerate(zip(teacher_token, student_token)):
            if t==s and t not in ignore_list and t.encode().isalpha():
                eng_mask[idx] = 1
        return eng_mask
    
    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        ## sentence ##
        x = self.x_data[idx]
        teacher_tokens = self.t_tokenizer.encode_plus(x, max_length=self.max_length, padding=self.padding,
                                                      return_tensors=self.return_tensors, truncation=True,
                                                      return_token_type_ids=self.return_token_type_ids,
                                                      return_attention_mask=self.return_attention_mask)
        
        student_tokens = self.s_tokenizer.encode_plus(x, max_length=self.max_length, padding=self.padding,
                                                      return_tensors=self.return_tensors, truncation=True,
                                                      return_token_type_ids=self.return_token_type_ids,
                                                      return_attention_mask=self.return_attention_mask)
        
        t_input_ids = teacher_tokens['input_ids'][0]
        t_att_mask = teacher_tokens['attention_mask'][0]
        
        s_input_ids = student_tokens['input_ids'][0]
        s_att_mask = student_tokens['attention_mask'][0]
        # token_type_ids = tokenizer_output.token_type_ids
        
        ## mask sampling ##
        eng_mask = self.get_eng_tokens_mask(x)
        
        ## label ##
        y = torch.tensor(self.y_data[idx], dtype=torch.long)
        # y = F.one_hot(y, num_classes=self.class_num).float()
        
        return (t_input_ids, t_att_mask), (s_input_ids, s_att_mask), eng_mask, y


def main():
    pass

if __name__ == "__main__":
    main()