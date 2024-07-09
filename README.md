# Metric Learning-based Specific Knowledge Distillation

## Introduction
- KM-BERT, a pre-trained language model based on the Korean language, is pre-trained on medical news, textbooks, and research papers, aiming to address downstream tasks in the medical domain.
- However, emergency department pediatric Electronic Medical Record (EMR) data primarily consist of free-text clinical notes with a substantial amount of code-switching.
- Simply fine-tuning KM-BERT may not be an effective approach.
- In this paper, we propose a robust model training method for free-text clinical notes.
    - Specific Knowledge Distillation based on Metric Learning

## Models
- **Student**
  - [Kor-Bert](https://huggingface.co/kykim/bert-kor-base)
  - [KM-Bert](https://github.com/KU-RIAS/KM-BERT-Korean-Medical-BERT)
- **Teacher**
  - [Bert-base](https://huggingface.co/bert-base-uncased)
  - [Multi-lingual Bert](https://huggingface.co/bert-base-multilingual-uncased)
  - [Clinical Bert](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)

## Our Method
![image](https://github.com/JoSangYeon/Audio_Classifier_Project/assets/28241676/74f61a8f-209c-419e-af3c-959a303b6737)


## run code
- `python main.py`

  ```bash
  # args #
  
  --random_seed
  	: default=17
  	: fix random seed
  --epochs 
  	: default=15
  	: train epochs
  --batch_size 
  	: default=64
  	: your batch size
  --method 
  	: default=base 
  	: train method : [base, mlkd, ensemble]
  --student 
  	: default=kobert # or kmbert
  	: student model (if method is [mlkd, ensemble])
  --teacher
  	: default=mbert # or bert, cbert
  	: teacher model (if method is [mlkd, ensemble])
  --model 
  	: default=kobert # or kmbert, bert, mbert, cbert
  	: if method is [base], fine-tuning code
  --vocab_type 
  	: default=S # or B
  	: if model is kmbert, select vocab size (small, big)
  --is_pt 
  	: default=False
  	: if model is kmbert, select re-pretrained model
  --bert_lr 
  	: default=1e-4
  	: bert body learning rate
  --classifier_lr
  	: default=1e-2
  	: classifier layer learning rate
  --alpha
  	: default=0.5
  	: hidn loss rate
  --beta
  	: default=0.1
  	: attn loss rate
  --max_seq_length
  	: default=384
  --padding
  	: default=max_length
  --cs_max_len
  	: default=32
  	: knowledge transfer count in each sampels
  --class_num
  	: default=2
  ```

### base model

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --random_seed 17 --epochs 10 --batch_size 64 --method base --model kmbert --vocab_type S --is_pt False
```

### MLKD model

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --random_seed 17 --epochs 10 --batch_size 64 --method mlkd --student kmbert --vocab_type S --is_pt False --teacher mbert
```

### MLKD-Ensemble model

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --random_seed 17 --epochs 10 --batch_size 64 --method ensemble --student kmbert --vocab_type S --is_pt False --teacher mbert
```

