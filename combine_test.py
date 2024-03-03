from combine_model import CombinedModel,MultiOutputDataset,train,eval
import numpy as np
import torch
import matplotlib.pyplot as plt
from combine_model import CombinedModel,MultiOutputDataset,train,eval,CombineUNet

x_test = np.load("F:\Wujiahui\deoniseProject\git_x.npy")
y_test_A = np.load("F:\Wujiahui\deoniseProject\git_y_A.npy")
y_test_B = np.load("F:\Wujiahui\deoniseProject\git_y_B.npy")
'''
x_test =np.load('F:\Wujiahui\deoniseProject\\valid.npy')

y_test_A = np.load('F:\Wujiahui\deoniseProject\\valid_x.npy')
y_test_B = np.load('F:\Wujiahui\deoniseProject\\valid_y.npy')
'''

x_test = torch.tensor(x_test,dtype=torch.float)
y_test_A = torch.tensor(y_test_A,dtype=torch.float)
y_test_B = torch.tensor(y_test_B,dtype=torch.float)
# Hyperparameters
input_dim = 1116
hidden_dim = 500
learning_rate = 0.001
num_epochs = 5
output_dim = 1
alpha = 0.5
num_components = 50
batch_size = 8

plt.ion()
index =0
x_test_1 = x_test[index]
y_test_1 = y_test_A[index]
y_test_2 = y_test_B[index]
plt.figure()
plt.plot(x_test_1)
plt.plot(y_test_1)

plt.pause(2)
#模型测试

torch.cuda.set_device(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_test_1 = x_test_1.to(device)
#model = UNet(in_ch=1,out_ch=1).to(device)
#model = model().to(device)
model = CombinedModel(input_dim=input_dim, hidden_dim=hidden_dim, num_components=num_components, output_dim=output_dim).to(device)#DenoisingCNN().to(device)
#model = CombineUNet(in_channels =1, out_channels=1,input_dim=1024,num_components=50,output_dim=1).to(device)
print(1)
model_save_path = r"F:\Wujiahui\deoniseProject\combine_model.pt"

model.load_state_dict(torch.load(model_save_path, map_location = device))
print(2)
model.eval()
print(3)
denoise_predict,reg_predict = model(x_test_1)
denoise_predict =denoise_predict.cpu().detach().numpy()
denoise_predict=denoise_predict.squeeze()

print(4)
plt.figure()
plt.plot(denoise_predict,label="y_test_predict1",linewidth=1.0)
plt.plot(y_test_1,label = 't_test',linewidth=1.0)
print("预测得到的浓度值为：",reg_predict)
print("实际的浓度值为：",y_test_2)
plt.ioff()
plt.show()
