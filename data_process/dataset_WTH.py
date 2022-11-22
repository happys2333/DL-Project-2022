import pandas as pd
from util.param import WTH_PATH
import locale
import numpy as np
import os
from torch.utils.data import Dataset, dataloader
import torch
from torch.utils.data import Dataset, DataLoader
from util.tools import StandardScaler
from util.timefeatures import time_features
import warnings
warnings.filterwarnings('ignore')

class WTH:
    def __init__(self):
        self.df = pd.read_csv(WTH_PATH)