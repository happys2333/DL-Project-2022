# S
python -u run.py --is_training 1 --root_path ../../dataset --data_path WTH.csv --model_id WTH_168_168 --model Autoformer --data custom --features S --seq_len 168 --label_len 168 --pred_len 168 --e_layers 2 --d_layers 1 --factor 3 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --itr 1 --target WetBulbCelsius --batch_size 8
python -u run.py --is_training 1 --root_path ../../dataset/ETT-small --data_path ETTh1.csv --model_id ETTh1_168_168 --model Autoformer --data ETTh1 --features S --seq_len 168 --label_len 168 --pred_len 168 --e_layers 2 --d_layers 1 --factor 3 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --itr 1 --target OT --batch_size 8
python -u run.py --is_training 1 --root_path ../../dataset/ETT-small --data_path ETTh1.csv --model_id ETTh1_96_192 --model Autoformer --data ETTh1 --features S --seq_len 96 --label_len 96 --pred_len 192 --e_layers 2 --d_layers 1 --factor 3 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --itr 2 --target OT --batch_size 16
python -u run.py --is_training 1 --root_path ../../dataset/ETT-small --data_path ETTh2.csv --model_id ETTh2_168_168 --model Autoformer --data ETTh2 --features S --seq_len 168 --label_len 168 --pred_len 168 --e_layers 2 --d_layers 1 --factor 3 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --itr 1 --target OT --batch_size 8


# M
python -u run.py   --is_training 1   --root_path ../../dataset/ETT-small/ --data_path ETTh1.csv  --model_id ETTh1_96_96  --model Autoformer  --data ETTh1 --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --itr 1 --batch_size 32

python -u run.py   --is_training 1   --root_path ../../dataset/ --data_path ECL.csv  --model_id ECL_96_96  --model Autoformer  --data custom --features M --seq_len 96 --label_len 48 --pred_len 96 --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --des 'Exp' --itr 2 --batch_size 16 --target MT_320