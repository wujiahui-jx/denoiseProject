from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import torch
import h5py
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import pickle
import torch.utils.data as Data
import matplotlib.pyplot as plt
import time
import random as rd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '2'
class DenoisingCNN(nn.Module):
    def __init__(self):
        super(DenoisingCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 512, 256)
        self.fc2 = nn.Linear(256, 4096)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 512)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x
class Model(nn.Module):     # 三层 hidden layer  2048-500-100-2048

    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features=1116, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=100)
        self.fc3 = nn.Linear(in_features = 100,out_features= 1000)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(in_features=1000, out_features=1116)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x= self.fc3(x)
        x = self.dropout1(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)
        return x

# 训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    for index,(batch_x,batch_y), in enumerate(train_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)

# 测试函数
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for  index,(batch_x,batch_y)in enumerate(test_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            output = model(batch_x)
            test_loss += criterion(output, batch_y).item()
    return test_loss / len(test_loader)

# K折交叉验证函数
def k_fold_cross_validation(model, dataset, criterion, optimizer, device, k=10, num_epochs=10, batch_size=4):
    kf = KFold(n_splits=k, shuffle=True)
    train_loss = 0
    test_loss = 0
    train_list = []
    test_list = []
    for fold, (train_index, test_index) in enumerate(kf.split(dataset)):
        train_dataset = torch.utils.data.Subset(dataset, train_index)
        test_dataset = torch.utils.data.Subset(dataset, test_index)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(num_epochs):
            train_loss = train(model, train_loader, criterion, optimizer, device)
            test_loss = test(model, test_loader, criterion, device)
            train_list.append(float(train_loss))
            test_list.append(float(test_loss))
            print(f"Fold [{fold+1}/{k}], Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")

    return train_list,test_list
# 加上损失曲线图的代码

def plot_loss_curve(train_losses, test_losses):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1,2'
    torch.cuda.set_device(2)
    train_data = np.load('F:\Wujiahui\deoniseProject\\train_data_CH4.npy')
    test_data = np.load('F:\Wujiahui\deoniseProject\\test_data_CH4.npy')
    #test_concertration = np.load('F:\Wujiahui\deoniseProject\\test_concertration.npy')
    x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(train_data, test_data, test_size=0.1)
    #_,_,y_train_concertration,y_test_concertration = train_test_split(train_data,test_concertration,test_size=0.1)
    np.save('F:\Wujiahui\deoniseProject\git_x.npy', x_test_1)
    np.save('F:\Wujiahui\deoniseProject\git_y.npy', y_test_1)
    np.save('F:\Wujiahui\deoniseProject\\train_x.npy', x_train_1)
    np.save('F:\Wujiahui\deoniseProject\\train_y.npy', y_train_1)

    """转为torch类型"""
    x_train_1 = torch.tensor(x_train_1, dtype=torch.float)
    x_test_1 = torch.tensor(x_test_1, dtype=torch.float)
    y_train_1 = torch.tensor(y_train_1, dtype=torch.float)
    y_test_1 = torch.tensor(y_test_1, dtype=torch.float)
    """准备数据集"""
    # data_set = Data_set(x_train_1, y_train_1, 50)
    # data_set_1 = Data_set(x_train_1, y_train_1, 100)
    # data_set_2 = Data_set(x_train_2, y_train_2, 100)
    num, seq = x_train_1.size()
    num1, seq1 = x_test_1.size()
    x_train_1 = torch.reshape(x_train_1, (num, 1, seq))
    x_test_1 = torch.reshape(x_test_1, (num1, 1, seq1))
    dataset = TensorDataset(x_train_1, y_train_1)  # 输入变量


    print(x_train_1.size())
    # 将训练集和验证集包括在dataset中
    model = Model().to(device)#DenoisingCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    train_losses = []
    test_losses = []
    train_loss,test_loss = k_fold_cross_validation(model, dataset, criterion, optimizer, device, k=10, num_epochs=50, batch_size=8)
    '''加载预训练权重'''
    #3 model_path = r"F:\Wujiahui\deoniseProject\model.pt"
    # filter_net.load_state_dict(torch.load(model_path))
    """保存模型"""
    model_save_path = r"F:\Wujiahui\deoniseProject\model.pt"
    torch.save(model.state_dict(), model_save_path)
    plot_loss_curve(train_losses, test_losses)





