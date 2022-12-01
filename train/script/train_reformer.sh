python -u run.py --is_training 1 --root_path ../../dataset --data_path WTH.csv --model_id WTH_336_168 --model Reformer --data custom --features S --seq_len 168 --label_len 168 --pred_len 168 --e_layers 2 --d_layers 1 --factor 3 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --itr 1 --target WetBulbCelsius --batch_size 8
python -u run.py --is_training 1 --root_path ../../dataset/ETT-small --data_path ETTh1.csv --model_id ETTh1_336_168 --model Reformer --data ETTh1 --features S --seq_len 168 --label_len 168 --pred_len 168 --e_layers 2 --d_layers 1 --factor 3 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --itr 1 --target OT --batch_size 8
python -u run.py --is_training 1 --root_path ../../dataset/ETT-small --data_path ETTh2.csv --model_id ETTh2_336_168 --model Reformer --data ETTh2 --features S --seq_len 168 --label_len 168 --pred_len 168 --e_layers 2 --d_layers 1 --factor 3 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --itr 1 --target OT --batch_size 8
python -u run.py --is_training 1 --root_path ../../dataset --data_path ECL.csv --model_id ECL_336_168 --model Reformer --data custom --features S --seq_len 168 --label_len 168 --pred_len 168 --e_layers 2 --d_layers 1 --factor 3 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --itr 1 --target MT_320 --batch_size 8