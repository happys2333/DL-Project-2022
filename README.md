# DL-Project-2022

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