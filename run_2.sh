CUDA_VISIBLE_DEVICES=1 python main.py --random_seed 17 --epochs 15 --batch_size 32 --method mlkd --student bert --teacher bio_mbert --alpha 0.6 --beta 0.2 --bert_lr 1e-4 --classifier_lr 1e-3
CUDA_VISIBLE_DEVICES=1 python main.py --random_seed 17 --epochs 15 --batch_size 32 --method mlkd --student bert --teacher roberta --alpha 0.6 --beta 0.2 --bert_lr 1e-4 --classifier_lr 1e-3
CUDA_VISIBLE_DEVICES=1 python main.py --random_seed 17 --epochs 15 --batch_size 32 --method mlkd --student bert --teacher bio_roberta --alpha 0.6 --beta 0.2 --bert_lr 1e-4 --classifier_lr 1e-3