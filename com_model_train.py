import scipy.io as scio
import numpy as np
import torch
import h5py
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from Unet import UNet
import torch.optim as optim
from combine_model import CombinedModel,MultiOutputDataset,Denoising1DCNN,Model,train,eval,WaveletTransformLayer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score


# 加载数据

X_train = np.load('F:\Wujiahui\deoniseProject\\train_data_CH4.npy')
x_target_denoise = np.load('F:\Wujiahui\deoniseProject\\test_data_CH4.npy')
y_target_regression = np.load('F:\Wujiahui\deoniseProject\\test_concertration.npy')
X_train =X_train[0:10000,:]
x_target_denoise=x_target_denoise[0:10000,:]
y_target_regression=y_target_regression[0:10000]
plt.figure()
plt.plot(y_target_regression)
plt.show()
#y_target_regression = np.load('F:\Wujiahui\deoniseProject\\test_concertration.npy')

x_train, x_test, y_train_A, y_test_A, y_train_B, y_test_B = train_test_split(X_train, x_target_denoise,y_target_regression, test_size=0.1)
np.save('F:\Wujiahui\deoniseProject\git_x.npy',x_test)
np.save('F:\Wujiahui\deoniseProject\git_y_A.npy',y_test_A)
np.save('F:\Wujiahui\deoniseProject\git_y_B.npy',y_test_B)

"""转为torch类型"""
x_train = torch.tensor(x_train,dtype=torch.float)
x_test = torch.tensor(x_test,dtype=torch.float)
y_train_A = torch.tensor(y_train_A,dtype=torch.float)
y_test_A = torch.tensor(y_test_A,dtype=torch.float)
y_train_B = torch.tensor(y_train_B,dtype=torch.float)
y_test_B = torch.tensor(y_test_B,dtype=torch.float)
"超参数"
# Hyperparameters
input_dim = 1116
hidden_dim = 500
learning_rate = 1e-4
num_epochs = 100
output_dim = 1
alpha = 0.6
num_components = 50
batch_size = 8

dataset = TensorDataset(x_train, y_train_A, y_train_B)
test_dataset = TensorDataset(x_test, y_test_A, y_test_B)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=False)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

"生成模型实例"
torch.cuda.set_device(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#odel = CombinedModel(input_dim=input_dim, hidden_dim=hidden_dim, num_components=num_components, output_dim=output_dim).to(device)
model = Model().to(device)
#model = WaveletTransformLayer(wavelet_name='db4', mode='zero',input_length=1116).to(device)
# Create model, loss criterion, and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

denoiseloss_list, regloss_list = train(model, train_loader, optimizer, criterion, num_epochs,alpha,device)
reg_list, relative_error = eval(model, test_loader, criterion,device)
"""保存模型"""
model_save_path = r"F:\Wujiahui\deoniseProject\combine_model.pt"
torch.save(model.state_dict(), model_save_path)
y_test_B = y_test_B.cpu().numpy().squeeze()
residuals = np.array(reg_list) - (y_test_B)
#R方系数

r2 = r2_score(y_test_B, reg_list)
print("预测得到的R2系数： ",r2)
relative_error_mean = np.mean(relative_error)
print("相对平均误差",relative_error_mean)
absolute_error = np.mean(abs(residuals))
print("绝对平均误差",absolute_error)
plt.ion()

plt.figure()
plt.plot(denoiseloss_list)
plt.plot(regloss_list)
plt.pause(2)

print(y_test_B.shape)
print(len(reg_list))
plt.figure()
plt.scatter(y_test_B,reg_list,s=8)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis("equal")
plt.axis("square")
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
plt.plot([0,0.1],[0,0.1],color = 'red')

plt.pause(2)
plt.figure()
plt.scatter(reg_list, residuals)
plt.xlabel('Predicted')
plt.ylabel('absolute_Residuals')
plt.axhline(y=0, color='red', linestyle='--')

plt.pause(2)
plt.figure()
plt.scatter(y_test_B,relative_error)
plt.xlabel('Predicted')
plt.ylabel('relative_Residuals')
plt.axhline(y=0, color='red', linestyle='--')



plt.ioff()
plt.show()


