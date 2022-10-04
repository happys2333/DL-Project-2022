import matplotlib as mt
import matplotlib.pyplot as plt
import numpy as np
import os

import pandas as pd

from param import ECL_PATH, ECL_PRED_INFORMER_PATH, ECL_PRED_AUTOFORMER_PATH


def ecl_show(pred_len=24, model="autoformer"):
    """
    Shows comparison between prediction of informer and ground truth of ECL's last sequence
    Firstly use ckpt of informer to predict a sequence, then use the prediction by setting 'ECL_RESULT_PATH'
    :return:
    """
    if model == "informer":
        result_path = ECL_PRED_INFORMER_PATH
    elif model == "autoformer":
        result_path = ECL_PRED_AUTOFORMER_PATH
    else:
        result_path = ECL_PRED_INFORMER_PATH

    ecl = pd.read_csv(ECL_PATH)

    ground_truth = np.array(ecl.MT_320.iloc[-pred_len:])
    pred = np.load(os.path.join(result_path, "real_prediction.npy"))
    target_pred = pred.squeeze()[:, -1]

    plt.plot(target_pred, label="Pred")
    plt.plot(ground_truth, label="GT")
    plt.legend()
    plt.show()


def prophet_show(pre_len=24, train_len=96):
    """
    Simply univariate prediction via Prophet
    :param pre_len: prediction length
    :param train_len: training data length
    :return:
    """
    try:
        from prophet import Prophet
    except Exception as e:
        print("No Prophet library")
        return

    ecl = pd.read_csv(ECL_PATH)
    mt320 = ecl.iloc[-(train_len + pre_len):, [0, -1]].rename(columns={"date": "ds", "MT_320": "y"})
    # print(mt320.head(5))

    m = Prophet()
    train_data = mt320.iloc[:-pre_len, :]
    m.fit(train_data)

    future = m.make_future_dataframe(periods=pre_len, freq="H")  # shape: (len(train_data) + pre_len, 2)
    # print(future.tail())

    forecast = m.predict(future)  # ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
    # print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    pred = np.array(forecast.yhat[-pre_len:])
    ground_truth = np.array(mt320.iloc[-pre_len:, 1])

    plt.plot(pred, label="Pred")
    plt.plot(ground_truth, label="GT")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    ecl_show()
    # prophet_show()
