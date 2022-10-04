import pandas as pd
from param import ECL_PATH, WTH_PATH, KDD_PATH, ETT_PATH_DIR
import locale
import numpy as np
import os


class ECL:
    def __init__(self):
        self.df = pd.read_csv(ECL_PATH)


class ETT:
    def __init__(self):
        self.df_h1 = np.array(pd.read_csv(ETT_PATH_DIR + "/ETTh1.csv"))
        self.df_h2 = np.array(pd.read_csv(ETT_PATH_DIR + "/ETTh2.csv"))
        self.df_m1 = np.array(pd.read_csv(ETT_PATH_DIR + "/ETTm1.csv"))
        self.de_m2 = np.array(pd.read_csv(ETT_PATH_DIR + "/ETTm2.csv"))
        

    def processed_data(self, arr):
        x_arr = []
        y_arr = []
        for i in range(len(arr)):
            x_arr.append(arr[i][0])
            y_arr.append(arr[i][1:])
        return np.array(x_arr), np.array(y_arr)


class WTH:
    def __init__(self):
        self.df = pd.read_csv(WTH_PATH)


class WTDATA:
    def __init__(self):
        self.df = pd.read_csv(KDD_PATH)


if __name__ == '__main__':
    E = ETT()
    print(E.processed_data(E.df_h1)[0])
    print(E.processed_data(E.df_h1)[1])
