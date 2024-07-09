CUDA_VISIBLE_DEVICES=1 python main.py --random_seed 17 --epochs 15 --batch_size 32 --method base --model kobert --bert_lr 1e-5 --classifier_lr 1e-2
CUDA_VISIBLE_DEVICES=1 python main.py --random_seed 17 --epochs 15 --batch_size 32 --method base --model bert --bert_lr 1e-5 --classifier_lr 1e-2
CUDA_VISIBLE_DEVICES=1 python main.py --random_seed 17 --epochs 15 --batch_size 32 --method base --model mbert --bert_lr 1e-5 --classifier_lr 1e-2
CUDA_VISIBLE_DEVICES=1 python main.py --random_seed 17 --epochs 15 --batch_size 32 --method base --model cbert --bert_lr 1e-5 --classifier_lr 1e-2
CUDA_VISIBLE_DEVICES=1 python main.py --random_seed 17 --epochs 15 --batch_size 32 --method base --model kmbert --vocab_type B --is_pt False --bert_lr 1e-5 --classifier_lr 1e-2
CUDA_VISIBLE_DEVICES=1 python main.py --random_seed 17 --epochs 15 --batch_size 32 --method base --model kmbert --vocab_type S --is_pt False --bert_lr 1e-5 --classifier_lr 1e-2
CUDA_VISIBLE_DEVICES=1 python main.py --random_seed 17 --epochs 15 --batch_size 32 --method base --model kmbert --vocab_type B --is_pt True --bert_lr 1e-5 --classifier_lr 1e-2
CUDA_VISIBLE_DEVICES=1 python main.py --random_seed 17 --epochs 15 --batch_size 32 --method base --model kmbert --vocab_type S --is_pt True --bert_lr 1e-5 --classifier_lr 1e-2