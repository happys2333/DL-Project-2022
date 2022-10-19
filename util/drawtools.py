import matplotlib.pyplot as plt
import numpy as np
from typing import Union
import pandas as pd

from util.dataset import *
from util.param import *


def plot_comparison(datasets: Union[list, tuple], models: Union[list, tuple], pred_len: int, features="S", save=False):
    for dataset in datasets:
        ground_truth = None
        for model in models:
            gt, pre = plot_pred(pred_len, dataset, model, features=features, show=False)
            assert gt is not None, pre is not None
            if ground_truth is None:
                ground_truth = gt
                plt.plot(gt, label="Ground Truth")
            plt.plot(pre, label="%s" % model)

        title_name = "Prediction with length %d on %s" % (pred_len, dataset)
        plt.title(title_name)
        plt.legend()
        plt.show()
        if save:
            plt.imsave(title_name)


def plot_pred(pred_len=24, dataset="ETTh1", model="informer", features="S", show=True):
    """
    Shows comparison between prediction of informer and ground truth of ECL's last sequence
    Firstly use ckpt of informer to predict a sequence, then use the prediction by setting 'ECL_RESULT_PATH'
    :return:
    """
    if dataset == "ETTh1":
        data = ETT().df_h1
        result_path = ETTH1_PRED[features][model]
    elif dataset == "ECL":
        data = ECL().df
        result_path = ECL_PRED[features][model]
    elif dataset == "WTH":
        data = WTH().df
        result_path = WTH_PRED[features][model]
    else:
        print("Dataset not found")
        return None, None
    if result_path is None:
        return None, None

    ground_truth = np.array(data.iloc[-pred_len:, -1])
    pred = np.load(os.path.join(result_path, "real_prediction.npy"))
    if features == "M":
        target_pred = pred.squeeze()[:, -1]
    else:
        target_pred = pred.squeeze()[:]

    if show:
        plt.plot(target_pred, label="Pred")
        plt.plot(ground_truth, label="GT")
        plt.legend()
        plt.show()

    return ground_truth, target_pred


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
    # plot_pred(192, "ETTh1", "informer", features="M")
    plot_comparison(datasets=["ECL"], models=["informer", "autoformer","reformer"], pred_len=168)
