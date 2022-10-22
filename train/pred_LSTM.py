import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, random_split, DataLoader
from sklearn.preprocessing import StandardScaler

from util.dataset import *
import tqdm
import os
import matplotlib.pyplot as plt
import argparse
import numpy as np


class LSTMDataSet(Dataset):
    def __init__(self, dataset_name, sequence_len, pred_len, feature_type="M", mode="train"):
        self.pred_len = pred_len
        self.seq_len = sequence_len
        self.total_len = self.pred_len + self.seq_len
        self.feature_type = -1 if feature_type == "S" else 0
        self.scaler = StandardScaler()
        self.mode = mode

        self.dataset = self.__get_dataset(dataset_name)

    def __get_dataset(self, dataset_name):
        """
        The final column is the default target
        """
        if dataset_name == "ECL":
            data = ECL().df.values[:, 1:].astype(float)
        elif dataset_name == "WTH":
            data = WTH().df.values[:, 1:].astype(float)
        elif dataset_name == "ETT":
            data = ETT().df_h1.values[:, 1:].astype(float)
        else:
            raise "No such dataset"

        data = data[:, self.feature_type:]
        # scale
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        data = torch.Tensor(data).to(torch.float32)

        N, D = data.shape
        train_N = N
        if self.mode == "train":
            train_N = int(N * 0.6)
            data = data[:train_N, :]
        elif self.mode == "test":
            train_N = N - int(N * 0.6)
            data = data[-train_N:, :]

        dataset_len = train_N - self.total_len + 1
        assert dataset_len > 0

        dataset = []
        for i in range(dataset_len):
            dataset.append(data[i:i + self.total_len, :])
        return dataset

    def __getitem__(self, item):
        source, target = self.dataset[item][:self.seq_len], self.dataset[item][self.seq_len:]
        return source, target

    def __len__(self):
        return len(self.dataset)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.dim_fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, h = self.lstm(x)
        x = self.dim_fc(lstm_out)

        return x


def train(dataset_name, input_size, hidden_size, output_size, pre_len, batch, epochs, early_stop_patience, feature_type,
          seq_len=None):
    if seq_len is None:
        seq_len = 3 * pre_len // 2
    assert seq_len >= pre_len

    # create folder
    ckpt_name = "%s_%s_is%d_hs%d_os%d_sl%d_pl%d" % (
        dataset_name, feature_type, input_size, hidden_size, output_size, seq_len, pre_len)
    ckpt_save_path = os.path.join(os.path.split(__file__)[0], "../ckpts/LSTM/%s" % ckpt_name)
    if not os.path.exists(ckpt_save_path):
        os.makedirs(ckpt_save_path)

    # hardware config
    device = "cup"
    if torch.cuda.is_available():
        device = "cuda:%d" % 0

    dataset = LSTMDataSet(dataset_name, sequence_len=seq_len, pred_len=pre_len, feature_type=feature_type)
    train_len = int(len(dataset) * 0.7)
    valid_len = len(dataset) - train_len
    train_data, valid_data = random_split(dataset, [train_len, valid_len])
    print(train_data)
    train_dataloader, valid_dataloader = DataLoader(train_data, batch), DataLoader(valid_data, batch)

    # build model and optimizer
    model = LSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss().to(device)

    early_stop_now = 0
    best_loss = torch.inf
    print("Training number: %d, validation number: %d", train_len, valid_len)
    for epoch in range(epochs):
        print("Epoch%d--------------------\nBegin training..." % epoch)
        bar = tqdm.tqdm(total=train_len // batch)
        for i, data in enumerate(train_dataloader):
            source, target = data[0].to(device), data[1].to(device)
            pred = model(source)[:, -pre_len:, :]

            optimizer.zero_grad()
            loss = criterion(target, pred)
            loss.backward()
            optimizer.step()
            bar.update()

        print("Begin validation...")
        total_loss = 0
        bar = tqdm.tqdm(total=valid_len // batch)
        for i, data in enumerate(valid_dataloader):
            source, target = data[0].to(device), data[1].to(device)
            pred = model(source)[:, -pre_len:, :]

            optimizer.zero_grad()
            loss = criterion(target, pred)
            loss.backward()
            optimizer.step()
            bar.update()

            total_loss += loss

        total_loss /= valid_len
        print("--->Validation Loss: %f" % float(total_loss))
        if best_loss > total_loss:
            print("Loss from %f -> %f, save model" % (best_loss, total_loss))
            best_loss = total_loss
            early_stop_now = 0
            torch.save(model.state_dict(), os.path.join(ckpt_save_path, ckpt_name + ".pt"))
        else:
            early_stop_now += 1
            print("early stop %d/%d" % (early_stop_now, early_stop_patience))
            if early_stop_now >= early_stop_patience:
                break


def pred_test(dataset_name, input_size, hidden_size, output_size, pre_len, feature_type, seq_len=None, index=-1,
              save_fig=False, save_data=False):
    if seq_len is None:
        seq_len = 3 * pre_len // 2
    assert seq_len >= pre_len

    dataset = LSTMDataSet(dataset_name, sequence_len=seq_len, pred_len=pre_len, feature_type=feature_type,
                          mode="pred_test")

    # find ckpt
    ckpt_name = "%s_%s_is%d_hs%d_os%d_sl%d_pl%d" % (
        dataset_name, feature_type, input_size, hidden_size, output_size, seq_len, pre_len)
    ckpt = os.path.join(os.path.split(__file__)[0], "../ckpts/LSTM/%s" % ckpt_name, ckpt_name + ".pt")

    # build model
    device = "cpu"
    model = LSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
    model.load_state_dict(torch.load(ckpt))

    # predict
    source, target = dataset[index][0].unsqueeze(0).to(device), dataset[index][1].unsqueeze(0).to(device)
    pred = model(source)[:, -pre_len:, :].squeeze(0).detach()
    gt = target[:, :, :].squeeze(0).detach()

    inverse_pred = dataset.scaler.inverse_transform(pred)
    inverse_gt = dataset.scaler.inverse_transform(gt)
    if save_data:
        np.save("real_prediction.npy", inverse_pred)
        np.save("true.npy", inverse_gt)

    pred = inverse_pred[..., -1:]
    gt = inverse_gt[..., -1:]

    plt.figure(figsize=(15, 5))
    plt.plot(pred, label="pred")
    plt.plot(gt, label="gt")
    plt.legend()
    if save_fig:
        plt.savefig("save.png")
    plt.show()


def test(dataset_name, input_size, hidden_size, output_size, pre_len, epochs, feature_type,
         seq_len=None):
    dataset = LSTMDataSet(dataset_name, sequence_len=seq_len, pred_len=pre_len, feature_type=feature_type,
                          mode="test")

    # find ckpt
    ckpt_name = "%s_%s_is%d_hs%d_os%d_sl%d_pl%d" % (
        dataset_name, feature_type, input_size, hidden_size, output_size, seq_len, pre_len)
    ckpt = os.path.join(os.path.split(__file__)[0], "../ckpts/LSTM/%s" % ckpt_name, ckpt_name + ".pt")

    # hardware config
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:%d" % 0

    # build model
    model = LSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
    model.load_state_dict(torch.load(ckpt))

    criterion_mse = nn.MSELoss().to(device)
    criterion_mae = nn.L1Loss(reduction="mean")

    dataset_len = len(dataset)
    mse = mae = 0
    print("Begin test %d data" % dataset_len)
    bar = tqdm.tqdm(total=dataset_len)
    for i in range(dataset_len):
        source, target = dataset[i]
        source = source.unsqueeze(0).to(device)
        target = target.unsqueeze(0).to(device)

        pred = model(source)[:, -pre_len:, :]
        loss_mse = criterion_mse(pred, target)
        loss_mae = criterion_mae(pred, target)
        mse += float(loss_mse)
        mae += float(loss_mae)
        bar.update()

    mse /= dataset_len
    mae /= dataset_len
    print("mse： %f, mae: %f" % (mse, mae))


def dataset_test():
    dataset = LSTMDataSet("ECL", seq_len, pre_len)
    print(len(dataset))
    print(len(dataset[0][0]), len(dataset[0][1]))


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='ECL', help='dataset, can be [ECL, ETT, WTH]')

parser.add_argument("--ipt_size", type=int, default=1, help='input size')
parser.add_argument("--hid_size", type=int, default=512)
parser.add_argument("--opt_size", type=int, default=1)
parser.add_argument("--pre_len", type=int, default=168)
parser.add_argument("--seq_len", type=int, default=168)
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--patience", type=int, default=3)
parser.add_argument("--feature_type", type=str, default="S")

parser.add_argument("--mode", type=str, default="train", help="train, test or pred_test")
parser.add_argument("--pred_idx", type=int, default=-1,
                    help="pred which group of data, a group is from [i:i+seq_len+pred_len], pred_idx==-1 means [-seq_len-pred_len:]")
parser.add_argument("--save_fig", action="store_true", help="for pred_test to save graph")
parser.add_argument("--save_data", action="store_true", help="for pred_test to save pred and true")

args = parser.parse_args()
if args.mode == "train":
    train(dataset_name=args.dataset, input_size=args.ipt_size, hidden_size=args.hid_size, output_size=args.opt_size,
          pre_len=args.pre_len, batch=args.batch, feature_type=args.feature_type, epochs=args.epochs,
          early_stop_patience=args.patience, seq_len=args.seq_len)
elif args.mode == "pred_test":
    pred_test(dataset_name=args.dataset, input_size=args.ipt_size, hidden_size=args.hid_size, output_size=args.opt_size,
              pre_len=args.pre_len, feature_type=args.feature_type, seq_len=args.seq_len, index=args.pred_idx,
              save_fig=args.save_fig, save_data=args.save_data)
else:
    test(dataset_name=args.dataset, input_size=args.ipt_size, hidden_size=args.hid_size, output_size=args.opt_size,
         pre_len=args.pre_len, feature_type=args.feature_type, epochs=args.epochs, seq_len=args.seq_len)
