import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, random_split, DataLoader
from sklearn.preprocessing import StandardScaler

from util.dataset import *
import tqdm
import os
import matplotlib.pyplot as plt


class LSTMDataSet(Dataset):
    def __init__(self, dataset_name, sequence_len, pred_len, feature_type="M"):
        self.pred_len = pred_len
        self.seq_len = sequence_len
        self.total_len = self.pred_len + self.seq_len
        self.feature_type = -1 if feature_type == "S" else 0
        self.scaler = StandardScaler()

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

        # scale
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        data = torch.Tensor(data).to(torch.float32)

        N, D = data.shape
        dataset_len = N - self.total_len + 1
        assert dataset_len > 0

        dataset = []
        for i in range(dataset_len):
            dataset.append(data[i:i + self.total_len, self.feature_type:])
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


def train():
    # dataset config
    dataset_name = "ETT"
    # network config
    batch = 16
    feature_type = "S"
    input_size = 1
    hidden_size = 512
    output_size = 1
    pre_len = 168
    seq_len = 3 * pre_len // 2
    assert seq_len >= pre_len

    # create folder
    ckpt_name = "%s_is%d_hs%d_os%d_sl%d_pl%d" % (
        dataset_name, input_size, hidden_size, output_size, seq_len, pre_len)
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
    train_dataloader, valid_dataloader = DataLoader(train_data, batch), DataLoader(valid_data, batch)

    # build model and optimizer
    model = LSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss().to(device)

    epochs = 100
    early_stop_patience = 3
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


def pred_test(index=-1):
    # dataset config
    dataset_name = "ECL"
    # network config
    batch = 16
    feature_type = "S"
    input_size = 1
    hidden_size = 512
    output_size = 1
    pre_len = 168
    seq_len = 3 * pre_len // 2
    assert seq_len >= pre_len

    dataset = LSTMDataSet(dataset_name, sequence_len=seq_len, pred_len=pre_len, feature_type=feature_type)

    # find ckpt
    ckpt_name = "%s_is%d_hs%d_os%d_sl%d_pl%d" % (
        dataset_name, input_size, hidden_size, output_size, seq_len, pre_len)
    ckpt = os.path.join(os.path.split(__file__)[0], "../ckpts/LSTM/%s" % ckpt_name, ckpt_name + ".pt")

    # build model
    device = "cpu"
    model = LSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
    model.load_state_dict(torch.load(ckpt))

    # predict
    source, target = dataset[index][0].unsqueeze(0).to(device), dataset[index][1].unsqueeze(0).to(device)
    pred = model(source)[:, -pre_len:, -1:]

    plt.plot(pred.squeeze().detach(), label="pred")
    plt.plot(target.squeeze().detach(), label="gt")
    plt.legend()
    plt.show()


def dataset_test():
    dataset = LSTMDataSet("ECL", seq_len, pre_len)
    print(len(dataset))
    print(len(dataset[0][0]), len(dataset[0][1]))


if __name__ == "__main__":
    train()
    # pred_test()
