import pandas as pd
from .param import ECL_PATH, WTH_PATH, KDD_PATH, ETT_PATH_DIR
import locale
from abc import abstractmethod, ABCMeta


class DataSet(metaclass=ABCMeta):
    @abstractmethod
    def train_data(self, args=None):
        raise NotImplementedError()


    @abstractmethod
    def vail_data(self, args=None):
        raise NotImplementedError()


    @abstractmethod
    def test_data(self, args=None):
        raise NotImplementedError()

    @abstractmethod
    def pred_data(self,args=None):
        raise NotImplementedError()


class ECL(DataSet):
    def __init__(self):
        self.df = pd.read_csv(ECL_PATH)

    def train_data(self, args=None):

        pass

    def vail_data(self, args=None):
        pass

    def test_data(self, args=None):
        pass

    def pred_data(self, args=None):
        pass


class ETT(DataSet):
    def __init__(self):
        self.df_h1 = pd.read_csv(ETT_PATH_DIR + "/ETTh1.csv")
        self.df_h2 = pd.read_csv(ETT_PATH_DIR + "/ETTh2.csv")
        self.df_m1 = pd.read_csv(ETT_PATH_DIR + "/ETTm1.csv")
        self.de_m2 = pd.read_csv(ETT_PATH_DIR + "/ETTm2.csv")


class WTH(DataSet):
    def __init__(self):
        self.df = pd.read_csv(WTH_PATH)


class WTDATA(DataSet):
    def __init__(self):
        self.df = pd.read_csv(KDD_PATH)


if __name__ == '__main__':
    E = ECL()
