import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import torch.utils.data as Data
import matplotlib.pyplot as plt
import time

from NeuralNetwork import Model,training,AttentionDenoisingModel,DenoiseModel
#from combine_model import CombinedModel,MultiOutputDataset,train,eval
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
train_data = np.load('F:\Wujiahui\deoniseProject\\train_data_CH4.npy')
test_data = np.load('F:\Wujiahui\deoniseProject\\test_data_CH4.npy')

plt.figure()
plt.plot(test_data[1])
plt.plot(test_data[10])
plt.plot(test_data[100])

plt.show()
#test_concertration = np.load("F:\Wujiahui\deoniseProject\\test_data_CH4.npy")

x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(train_data, test_data, test_size=0.1, random_state=2)



np.save('F:\Wujiahui\deoniseProject\git_x.npy',x_test_1)
np.save('F:\Wujiahui\deoniseProject\git_y.npy',y_test_1)
np.save('F:\Wujiahui\deoniseProject\\train_x.npy',x_train_1)
np.save('F:\Wujiahui\deoniseProject\\train_y_y.npy',y_train_1)



"""转为torch类型"""
x_train_1 = torch.tensor(x_train_1,dtype=torch.float)
x_test_1 = torch.tensor(x_test_1,dtype=torch.float)
y_train_1 = torch.tensor(y_train_1,dtype=torch.float)
y_test_1 = torch.tensor(y_test_1,dtype=torch.float)




plt.figure()
plt.plot(x_train_1[99])
plt.plot(y_train_1[99])
plt.show()
"""准备数据集"""
#data_set = Data_set(x_train_1, y_train_1, 50)
# data_set_1 = Data_set(x_train_1, y_train_1, 100)
# data_set_2 = Data_set(x_train_2, y_train_2, 100)
num, seq = x_train_1.size()
num1,seq1 = x_test_1.size()
x_train_1 = torch.reshape(x_train_1, (num, 1, seq))
#x_test_1 = torch.reshape(x_test_1, (num1, 1, seq1))
data_set = TensorDataset(x_train_1,y_train_1)  # 输入变量

batch_size=16
data_set = DataLoader(dataset=data_set, batch_size=16, shuffle=True, drop_last=False)
"""生成模型实例"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(2)
#model = Model().to(device)
#model = AttentionDenoisingModel(1116).to(device)
model = DenoiseModel().to(device)
"""定义criterion, optimizer"""
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
'''加载预训练权重'''
#model_path = r"F:\Wujiahui\deoniseProject\model.pt"
#filter_net.load_state_dict(torch.load(model_path))
"""训练模型"""
begin_time = time.time()
train_loss, test_loss = training(model, data_set, x_test_1, y_test_1, device, optimizer, criterion, epochs=100)

#train_loss,test_loss = train(model,data_set,optimizer,criterion,device,epoch=1000)
end_time = time.time()
total_time_cost = (end_time - begin_time) / 60
print("总训练用时：{} 分钟".format(total_time_cost))
"""保存模型"""
model_save_path = r"F:\Wujiahui\deoniseProject\model.pt"
torch.save(model.state_dict(), model_save_path)

'''绘制train_Loss'''

plt.figure()
plt.plot(train_loss, label="train loss")
plt.plot(test_loss,label = 'test_label')
plt.show()


