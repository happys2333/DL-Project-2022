#!/bin/bash
# run it in root directory

# S
python train/pred_LSTM.py --dataset ECL --ipt_size 1 --opt_size 1 --seq_len 168 --pre_len 168 --batch 16 --feature_type S --mode train
python train/pred_LSTM.py --dataset ETT --ipt_size 1 --opt_size 1 --seq_len 168 --pre_len 168 --batch 16 --feature_type S --mode train
python train/pred_LSTM.py --dataset WTH --ipt_size 1 --opt_size 1 --seq_len 168 --pre_len 168 --batch 16 --feature_type S --mode train

# M
python train/pred_LSTM.py --dataset ECL --ipt_size 321 --opt_size 321 --seq_len 192 --pre_len 192 --batch 16 --feature_type M --mode train
python train/pred_LSTM.py --dataset WTH --ipt_size 12 --opt_size 12 --seq_len 192 --pre_len 192 --batch 16 --feature_type M --mode train
python train/pred_LSTM.py --dataset ETT --ipt_size 7 --opt_size 7 --seq_len 192 --pre_len 192 --batch 16 --feature_type M --mode train