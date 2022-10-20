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


def draw_result_bar():
    result_path = "../result/multivariate/sq96_lb48_pre192_mul_wth.csv"
    result = pd.read_csv(result_path)

    size = 2

    x = np.arange(size)

    # 有a/b两种类型的数据，n设置为2
    total_width, n = 0.6, 3
    # 每种类型的柱状图宽度
    width = total_width / n

    data = [[0.542752, 0.525745],
            [0.587715, 0.559554],
            [0.745868, 0.622514]]
    colors = ["#4E79A7", "#A0CBE8", "#F28E2B", "#FFBE7D", "#59A14F", "#8CD17D", "#B6992D", "#F1CE63", "#499894",
              "#86BCB6", "#E15759", "#E19D9A"]
    # 重新设置x轴的坐标
    x = x - (total_width - width) / n
    # 画柱状图
    plt.bar(x, data[0], width=width, label="Informer", color=colors[0])
    plt.bar(x + width, data[1], width=width, label="Autoformer", color=colors[1])
    plt.bar(x + 2 * width, data[2], width=width, label="LSTM", color=colors[2])
    plt.xticks(np.arange(size), ("MSE", "MAE"))
    # 显示图例
    # plt.figure(dpi=300,figsize=(24,24))
    plt.legend(loc='lower right', prop={"family": "Times New Roman"})
    plt.xlabel("Metric Comparison", fontname="Times New Roman")
    plt.ylabel("Value of MSE or MAE", fontname="Times New Roman")
    # 显示柱状图
    plt.show()


if __name__ == "__main__":
    # plot_pred(168, "ETTh1", "informer", features="S")
    draw_result_bar()
    # plot_comparison(datasets=["ECL"], models=["informer", "autoformer","reformer"], pred_len=168)
