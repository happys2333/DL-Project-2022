import matplotlib as mt
import matplotlib.pyplot as plt
import numpy as np
import os

import pandas as pd

from param import ECL_PATH, ECL_PRED_PATH


def ecl_show(pred_len=24):
    """
    Shows comparison between prediction of informer and ground truth of ECL's last sequence
    Firstly use ckpt of informer to predict a sequence, then use the prediction by setting 'ECL_RESULT_PATH'
    :return:
    """
    ecl = pd.read_csv(ECL_PATH)

    ground_truth = np.array(ecl.MT_320.iloc[-pred_len:])
    pred = np.load(os.path.join(ECL_PRED_PATH, "real_prediction.npy"))
    target_pred = pred.squeeze()[:, -1]

    plt.plot(target_pred, label="Pred")
    plt.plot(ground_truth, label="GT")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    ecl_show()
