from combine_model import CombinedModel,MultiOutputDataset,train,eval
import numpy as np
import torch
import matplotlib.pyplot as plt
from combine_model import CombinedModel,MultiOutputDataset,CombineCNN,train,eval,CombineUNet,WaveletTransformLayer,Model
x_test = np.load("F:\Wujiahui\deoniseProject\git_x.npy")
y_test_A = np.load("F:\Wujiahui\deoniseProject\git_y_A.npy")
y_test_B = np.load("F:\Wujiahui\deoniseProject\git_y_B.npy")

'''
x_test = x_test[:,:1024]
y_test_A =y_test_A[:,:1024]
y_test_B =y_test_B[:,:1024]
'''
x_test = torch.tensor(x_test,dtype=torch.float)
y_test_A = torch.tensor(y_test_A,dtype=torch.float)
y_test_B = torch.tensor(y_test_B,dtype=torch.float)
"""Unet模型时使用"""
num, seq = x_test.size()
x_test= torch.reshape(x_test, (num, 1, seq))
y_test_A =torch.reshape(y_test_A,(num,1,seq))


# Hyperparameters
input_dim = 1116
hidden_dim = 500
learning_rate = 0.001
num_epochs = 5
output_dim = 1
alpha = 0.2
num_components = 50
batch_size = 8

plt.ion()
index =300
x_test_1 = x_test[index].unsqueeze(0)
y_test_1 = y_test_A[index].unsqueeze(0)
y_test_2 = y_test_B[index].unsqueeze(0)
plt.figure()
plt.plot(x_test_1.squeeze())
plt.plot(y_test_1.squeeze())
plt.pause(2)
#模型测试

torch.cuda.set_device(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_test_1 = x_test_1.to(device)
#model = UNet(in_ch=1,out_ch=1).to(device)
#model = model().to(device)
model = Model().to(device)#DenoisingCNN().to(device)
#model = CombineUNet(in_channels =1, out_channels=1,input_dim=1024,num_components=50,output_dim=1).to(device)
#model = CombineCNN(input_dim=1024, num_components=50,output_dim=1).to(device)
#model = WaveletTransformLayer(wavelet_name='db4', mode='zero',input_length=1116).to(device)

print(1)
model_save_path = r"F:\Wujiahui\deoniseProject\combine_model.pt"

model.load_state_dict(torch.load(model_save_path, map_location = device))
print(2)
model.eval()
print(3)
denoise_predict,reg_predict = model(x_test_1)
denoise_predict =denoise_predict.cpu().detach().numpy()
denoise_predict=denoise_predict.squeeze()
y_test_1= y_test_1.squeeze()
print(4)
plt.figure()

plt.plot(y_test_1,label = 't_test',linewidth=1.0)
plt.plot(denoise_predict,label="y_test_predict1",linewidth=1.0)

plt.ioff()
plt.show()
print("预测得到的浓度值为：",reg_predict)
print("实际的浓度值为：",y_test_2)