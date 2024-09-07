# DSG-KD_ Knowledge Distillation from Domain-Specific to General Language Models

## Introduction
The utilization of pre-trained language models, fine-tuned to address specific downstream tasks, is a common approach in natural language processing (NLP). However, acquiring domain-specific knowledge through fine-tuning alone is challenging. Traditional methods involve pre-training language models with vast amounts of domain-specific data before fine-tuning for particular tasks. This paper investigates the emergency/non-emergency classification task using Electronic Medical Record (EMR) data from a pediatric emergency department (PED) in Korea. Our findings reveal that existing domain-specific pre-trained language models underperform compared to general language models in handling N-Lingual free-text data characteristic of non-English speaking regions. Motivated by these limitations, we propose the Domain Knowledge Transfer methodology, which leverages Knowledge Distillation (KD) to infuse general language models with domain-specific knowledge during fine-tuning.

This study demonstrates the effective transfer of specialized knowledge between models by defining a general language model as the student and a domain-specific pre-trained model as the teacher. We specifically address the complexities of PED’s EMR data in non-English speaking environments, such as Korea, and show that our method enhances classification performance in these contexts. Our methodology not only outperforms baseline models on Korean PED EMR data but also suggests broader applicability across various professional and technical domains. Future work will focus on extending this methodology to include diverse non-English speaking regions and additional downstream tasks, aiming to develop advanced model architectures through the latest KD techniques.

## Our Method
![figure_1 - model_architecture](https://github.com/JoSangYeon/DSG-KD/assets/28241676/4cd2f0ea-6888-41ac-bff2-4b4761ad25c7)


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

