# S
python -u main_informer.py --model informer --data ECL --root_path ../../dataset/ --features S --freq h --seq_len 168 --label_len 168 --pred_len 336 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 1 --batch_size 4

python -u main_informer.py --model winformer --data ETTh1 --root_path ../../dataset/ETT-small --features S --freq h --seq_len 168 --label_len 168 --pred_len 168 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 1 --batch_size 16


# M
python -u main_informer.py --model winformer --data ETTh1 --root_path ../../dataset/ETT-small --features M --freq h --seq_len 96 --label_len 48 --pred_len 192 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 1 --batch_size 16

python -u main_informer.py --model winformer --data ECL --root_path ../../dataset --features M --freq h --seq_len 96 --label_len 48 --pred_len 192 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 1 --batch_size 32

python -u main_informer.py --model winformer --data WTH --root_path ../../dataset --features M --freq h --seq_len 96 --label_len 48 --pred_len 192 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 1 --batch_size 16