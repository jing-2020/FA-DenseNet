# coding=utf-8

from scipy.io import loadmat
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sys
from math import ceil
import optuna

torch.manual_seed(88)
torch.cuda.manual_seed(88)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class block(nn.Module):
    def __init__(self, in_features, hidden1, hidden2, hidden3, out_features):
        super(block, self).__init__()
        self.short = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.ReLU()
        )
        self.density = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Linear(hidden3, out_features)
            , nn.Dropout(0.5)
        )

    def forward(self, x):
        x = self.density(self.short(x))
        return x


class Fullnet(nn.Module):
    def __init__(self, in_features, out_features,a,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15):
        super(Fullnet, self).__init__()
        self.input = nn.Linear(in_features, a)
        self.block1 = block(a, a1, a2, a3, a)
        self.block2 = block(a, a4, a5, a6, a)
        self.block3 = block(a, a7, a8, a9, a)
        self.block4 = block(a, a10, a11, a12, a)
        self.block5 = block(a, a13, a14, a15, out_features)
        self.output = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.input(x)
        d1 = x
        x = self.block1(d1)
        d2 = x + d1
        x = self.block2(d2)
        d3 = x + d1 + d2
        x = self.block3(d3)
        d4 = x + d1 + d2 + d3
        x = self.block4(d4)
        d5 = x + d1 + d2 + d3 + d4
        x = self.block5(d5)
        x = self.output(x)
        return x


class mydata(Dataset):
    def __init__(self, file1, file2):
        data=loadmat(file1)
        data=data['P_abnormal']
        q=pd.read_excel(file2,header=None)
        q=q.values
    
        data=data[:,[447, 314, 276, 327, 153, 111, 230,   0, 422,  79, 351,  75, 288, 410, 355, 365, 396, 160, 131, 456, 260,  34, 209,  55]]
        label=np.zeros((data.shape[0],1),dtype=np.int64)
        for j,i in enumerate(range(0,data.shape[0],24)):
            label[i:(i+24),0]=q[j,0]-1
        model=MinMaxScaler()
        data=model.fit_transform(data)
        self.data = data
        self.data = self.data.astype(np.float32)
        self.label=label


    def __getitem__(self, index):
        
        return self.data[index, :], self.label[index, 0] - 1

    def __len__(self):
        return self.data.shape[0]


def top5(precision, true, device):
    value, index = torch.topk(precision, 5, dim=1)
    numbers = true.shape[0]
    accuracy = torch.zeros(numbers).to(device)
    for i in range(numbers):
        if true[i] in index[i, :]:
            accuracy[i] = 1
    return (torch.sum(accuracy) / torch.Tensor([numbers]).to(device)).item()


def top1(precision, true, device):
    index = torch.max(precision, 1)[1]
    accuracy = sum(index == true) / torch.Tensor([true.shape[0]]).to(device)
    return accuracy.item()

def objective(trial):

    train_size = 0.8
    BATCH_SIZE = 320
    EPOCH = 100
    LR = 0.001
    a=trial.suggest_int("a",40,80)
    a1=trial.suggest_int("a1",80,160)
    a2 = trial.suggest_int("a2", 160, 320)
    a3 = trial.suggest_int("a3", 80, 160)
    a4=trial.suggest_int("a4",80,200)
    a5 = trial.suggest_int("a5", 160, 480)
    a6 = trial.suggest_int("a6", 80, 200)
    a7=trial.suggest_int("a7",80,160)
    a8 = trial.suggest_int("a8", 160, 320)
    a9 = trial.suggest_int("a9", 80, 160)
    a10=trial.suggest_int("a10",80,160)
    a11 = trial.suggest_int("a11", 160, 320)
    a12 = trial.suggest_int("a12", 80, 160)
    a13=trial.suggest_int("a13",80,160)
    a14 = trial.suggest_int("a14", 160, 320)
    a15 = trial.suggest_int("a15", 80, 160)


    Data = mydata(r"data.mat", r'area_id.xlsx')
    train_size = int(len(Data) * train_size)
    test_size = len(Data) - train_size
    train_data, test_data = random_split(Data, [train_size, test_size])
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=len(test_data))

    model = Fullnet(24, 24,a,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    F_loss = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for step, [x, y] in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            py = model(x)
            loss = F_loss(py, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            for tx, ty in test_loader:
                tx, ty = tx.to(device), ty.to(device)
                pty = model(tx)
                loss = F_loss(pty, ty)
                accuracy_top1 = top1(pty, ty, device)
                # accuracy_top5 = top5(pty, ty, device)
            model.train()
        trial.report(accuracy_top1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy_top1

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    # print("  Params: ")
    # for key, value in trial.params.items():
        # print("    {}: {}".format(key, value))