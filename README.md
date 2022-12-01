# DL-Project-2022

## Experiments
### Wformer

#### 中期答辩没答上来的问题

老师的问题：降采样用了卷积，卷积需要固定向量长度，这样导致需要固定输入长度模型才能运行（obsismc's理解）；或者说test的数据要是和training的数据长度不一样就无法运行

解释：

没这个问题

1. 卷积核大小是1和长度无关，只是为了投影到其他空间；maxpooling才是降采样的功能，所以不用怕输入长度不一样导致无法下采样
2. 或许会问上采样的时候怎么办，上采样似乎也和向量长度有关，而且每一层的向量长度都不一样。回答：这种情况Linear层确实无法上采样，
    因为Linear层需要固定的输入和输出长度，但是我用反卷积。反卷积对输入长度无限制，但是在我这个模型中对输出长度有要求，最下层的长度是次下层长度的一半，所以反卷积之后需要把长度变成两倍，这里参考反卷积公式
  
   $L_{out} = (L_{in}-1)\times stride - 2\times padding + kernel\_size + output\_padding$
   
   如果想要输出长度是输入的两倍且和输入长度无关，则stride必须是2，其他参数满足等式即可
   
3. transformer本身不受数据长度限制，这个问题和transformer自身架构无关



#### 思考

1. Speed of Converge improves?
2. Model doesn't matter but structure matters? Change the model in Wformer
3. Need to check the actual graph instead of MSE or MAE
4. some cases overfitted?
5. influenced by sequence length and label length? is its learning skill better?
6. 和autoformer的趋势差不多？当autoformer比informer好，wformer也好？
7. 多变量不行
8. 可以借鉴Yformer将encoder和decoder结合
9. 现在只有时间点和时间段的融合（广度），没有像U-Net一样做深度上的结合
10. 改进点：不同层的注意力加权，第一层用提取段，后面提取点？
11. 深（广）度一直到无法继续除以2？

## Reference Model
### Informer2020
#### Train
```shell
# ECL example
python -u main_informer.py --model informer --data ECL --attn prob --freq h --root_path E:\Data\GitRepo\DL-Project-2022\dataset --data_path ECL_without_last24.csv --do_predict --inverse
```
Use `--do_predict` to make prediction.

Pay attention: result is normalized, if you need original data, `--inverse` should be added
#### Predict
It uses the last span of sequence of data to predict unknown future sequence.

Related function is `exp.predict()` in `main_informer.py`

For simplicity, we wrote a `predict_informer.py` in `submodule/utils` to help us make predictions.
```shell
python -u predict_informer.py --model informer --data ECL --attn prob --freq h --root_path E:\Data\GitRepo\DL-Project-2022\dataset --data_path ECL_without_last24.csv --idx 0 --inverse
```
#### *Result* folder
`metrics.npy`: [mae, mse, rmse, mape, msp]

### Autoformer
The usage is almost the same as *Informer*, just refer to procedure above.

Training script refers to `script` folder, like
```shell
# script/ECL_script/Autoformer.sh
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model Autoformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1
  
  python -u run.py --is_training 1 --root_path ../../dataset --data_path ECL.csv --model_id ECL_168_168 --model Autoformer --data custom --features S --seq_len 168 --label_len 168 --pred_len 168 --e_layers 2 --d_layers 1 --factor 3 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --itr 1 --target MT_320 --batch_size 8

```

As for prediction, up to now (2022.10.5) we haven't found a simple way to denormalize 
the output, so we modify original code like following:

```python
# data_provider/data_loader.py, make inverse True
class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=True, timeenc=0, freq='15min', cols=None):

# exp/exp_main.py, in predict method (about 305 line), add inverse_transform method
<< << << origin
else:
if self.parser.output_attention:
    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
else:
    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
pred = outputs.detach().cpu().numpy()  # .squeeze()
preds.append(pred)
== == == == == ==
else:
if self.parser.output_attention:
    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
else:
    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
outputs = pred_data.inverse_transform(outputs.cpu().squeeze(0))
pred = outputs  # .squeeze()
>> >> >> modified
```
Then use `pred_autoformer.py` whose usage please refer to *informer*, shell:
```shell
python -u pred_autoformer.py --root_path ./dataset/electricity/ --data_path electricity.csv --model_id ECL_96_96 --model Autoformer --data custom --seq_len 96 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --des 'Exp' --idx 0
```

### Prophet
Details: https://github.com/facebook/prophet

### DeepAR
With assistance of [`pytorch-forecasting`](https://github.com/jdb78/pytorch-forecasting) library and 
its tutorial of [DeepAR](https://pytorch-forecasting.readthedocs.io/en/latest/tutorials/deepar.html)

Need to install `pytorch-lightning` and `pytorch-forecasting`.

Related code is `train/pred_DeepAR.ipynb`.However, the result isn't good.

### LSTM
We just use `nn.LSTM` in `pytorch`. Out of our expectation, its performance isn't bad.
## Dataset
ETT, Weather, ECL,  [Traffic](https://archive.ics.uci.edu/ml/datasets/PEMS-SF)