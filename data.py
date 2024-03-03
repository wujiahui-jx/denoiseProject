import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import math
matplotlib.rc("font", family='Microsoft YaHei')

def awgn(signal,SNR):
    a = signal-1
    sigpower = np.sum(np.abs(a)**2)/np.size(a)
    noisepower = sigpower/(10**(SNR/10))
    noise = np.sqrt(noisepower)*np.random.randn(np.size(a))
    y = signal+noise
    return y
df = pd.read_csv('1%_CH4.csv')
data = np.array(df)
R = 0.001
n=1.5
L = 1.3
lamada = data[:,0]
fia = 1*np.pi*n*L/lamada
It = 1/(1+(4.0 * R * pow(np.sin(fia/2.0),2))/pow(1-R,2))

'''IGMs_gas = data[:, 1]
IGMs_gas_ideal = data[:,1]
x=19000-4096
IGMs_gas = IGMs_gas[x:19000]
IGMs_gas_ideal = IGMs_gas_ideal[x:19000]
plt.figure()
plt.plot(np.exp(-2*IGMs_gas))
plt.plot(np.exp(-0.5*IGMs_gas))
plt.title('HCN吸光度曲线')
plt.show()
'''
IGMs_gas = data[:, 1]
x = data[:,0]
plt.figure()
plt.plot(x,IGMs_gas)
plt.ylabel("Aborbance")
plt.xlabel("Wavelength")

plt.show()

length = len(IGMs_gas)
interval = np.linspace(0.1e-2,0.1,10000)
#interval1 = np.linspace(0.01,0.1,10000)#np.linspace(0.5,2,2000)
#interval = np.hstack((interval,interval1))

con_num = len(interval)
test_data = np.zeros([con_num,length])
test_concertration = np.hstack((np.linspace(10,100,10000),np.linspace(100,1000,10000)))/1000

for i in range(con_num):
    test_data[i,:] = interval[i] * IGMs_gas

noise = np.zeros([1,length])
#test_data = np.tile(test_data,[5,1]) # 每组浓度相同的个数
#test_concertration = np.tile(test_concertration,[5,1])
total =len(test_data)
train_data = np.zeros([total,length])



plt.figure()
plt.plot(test_data[0,:])
plt.plot(test_data[99,:])
plt.plot(test_data[999,:])
plt.plot(test_data[9999,:])


plt.title('CH4吸光度曲线')
plt.show()
for i in range(total):
    noise = (9.71886e-5**0.5)*np.random.randn(1,length)* It
    train_data[i,:] = test_data[i,:] + noise
    noise = np.zeros([1, length])
train_data = np.exp(-train_data)
test_data = np.exp(-test_data)


plt.figure()
plt.plot(train_data[1999,:])
plt.plot(test_data[1999,:])
plt.title('透射光谱')
plt.show()

plt.figure()
plt.plot(test_concertration)
plt.show()

def SNR(origal,noise_signal):
    noise = noise_signal-origal
    sigpower = np.sum(np.abs(origal)**2)/np.size(origal)
    nopower = np.sum(np.abs(noise)**2)/np.size(noise)
    snr = 10*np.log10(sigpower/nopower)
    return snr


def compute_snr(pure_signal, noisy_signal):
    signal_to_noise_ratio = 10 * (np.log10(np.std(pure_signal) / np.std(noisy_signal - pure_signal)))
    return signal_to_noise_ratio



'''valid0 = 0.5*0.5e-1*np.random.randn(1,length)* It + data[:,1]
valid1 = 0.5e-1*np.random.randn(1,length)* It + data[:,1]
valid2 = 1.5*0.5*0.5e-1*np.random.randn(1,length)* It + data[:,1]
snr0 = SNR(data[:,1],valid0)
snr1 = SNR(data[:,1],valid1)
snr2 = SNR(data[:,1],valid2)
print(snr0)
print(snr1)
print(snr2)
valid0 = np.exp(-valid0)
valid1 = np.exp(-valid1)
valid2 = np.exp(-valid2)
real = np.exp(-data[:,1])
plt.ion()
plt.figure()
plt.plot(valid0.squeeze())
plt.plot(real)
plt.figure()
plt.plot(valid1.squeeze())
plt.plot(real)
plt.figure()
plt.plot(valid2.squeeze())
plt.plot(real)
plt.show()
valid =np.vstack((valid0,valid1,valid2))
valid_x = np.vstack((real,real,real))
valid_y = np.vstack((0.05,0.05,0.05))
np.save('F:\Wujiahui\deoniseProject\\valid.npy',valid)
np.save('F:\Wujiahui\deoniseProject\\valid_x.npy',valid_x)
np.save('F:\Wujiahui\deoniseProject\\valid_y.npy',valid_y)
'''
plt.figure()
plt.plot(np.std(train_data,axis=1))
plt.show()
plt.figure()
plt.plot(train_data[0])
plt.plot(test_data[0])
plt.plot(train_data[9999])
plt.plot(test_data[9999])

plt.show()
snr1= compute_snr(test_data[0],train_data[0])
snr2= compute_snr(test_data[9999],train_data[9999])
print(snr1)
print(snr2)


'''
# 使用小波去噪
import pywt
x = train_data[13000]
y = test_data[13000]
coeffs = pywt.wavedec(x, 'db4',level =7)  # 使用Daubechies小波系列
coeffs_thresholded = [pywt.threshold(coeff, value=0.033, mode='soft') for coeff in coeffs]
print(coeffs[0])
y_denoised = pywt.waverec(coeffs_thresholded, 'db4')

# 绘制原始信号和去噪后的信号
plt.figure(figsize=(12, 6))

plt.plot( x, label='Noisy signal')
plt.plot(y)
plt.plot( y_denoised, 'r', label='Denoised signal')
plt.legend()

plt.figure()
plt.plot(y)
plt.plot(y_denoised)
plt.show()



np.save('F:\Wujiahui\deoniseProject\\test_data_CH4.npy',test_data)
np.save('F:\Wujiahui\deoniseProject\\train_data_CH4.npy',train_data)
np.save('F:\Wujiahui\deoniseProject\\test_concertration.npy',test_concertration)


'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

pre_0 = np.load('Pre_DNSR_re0.npy')
pre_1 = np.load('Pre_DNSR_re1.npy')
pre = np.load('Pre_DNSR.npy')
x = np.min(pre_0,axis=1)
y_test_1 = np.load('F:\Wujiahui\deoniseProject\git_y.npy')
x_test_1 = np.load('F:\Wujiahui\deoniseProject\git_x.npy')
y_test_2 = np.load('F:\Wujiahui\deoniseProject\git_y_B.npy')
print(pre_0.shape)
print(pre_1.shape)

index = 89
plt.figure()
#plt.plot(x_test_1[index])
plt.plot(y_test_1[index])
plt.plot(pre_0[index])
plt.savefig('1.png')
a = np.min(pre,axis=1)
b = np.min(y_test_1,axis=1)
print(np.mean(abs(b-a)))

print(abs(pre_1[index]-y_test_2[index]))
print(abs(pre_1[index]-y_test_2[index])/y_test_2[index])
R = r2_score(x,b)
print(R)

plt.scatter(a,b,s=4)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis("equal")
plt.axis("square")
plt.xlim([0.95,plt.xlim()[1]])
plt.ylim([0.95,plt.ylim()[1]])
plt.plot([0.95,1],[0.95,1],color = 'red')
plt.savefig('2.png')
plt.close('all')

i = np.min(y_test_1,axis=1)
print(np.argmin(i))
index = 89
plt.figure()
plt.subplot(211)
plt.plot(x_test_1[index],label="label")
plt.plot(y_test_2[index],label='label')
plt.plot(pre_0[index],label="predict")

plt.legend(['predict','label'],fontsize= '18')

plt.subplot(212)
plt.plot(y_test_1[index]-pre_0[index],label="残差")

plt.legend(['residual'],fontsize= '18')
plt.savefig('1.png')

r2 = r2_score(y_test_2,pre)
print(r2)

plt.figure()
plt.scatter(y_test_2,pre_1,s=8)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis("equal")
plt.axis("square")
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
plt.plot([0,1],[0,1],color = 'red')
plt.savefig('2.png')
plt.close('all')

plt.plot(i)
plt.savefig('i.png')

pre =np.squeeze(pre)
pre_1 =np.squeeze(pre_1)
absulute = abs(pre_1-y_test_2)
print(np.argmax(absulute))
relative = abs(pre_1-y_test_2)/y_test_2
print(np.mean(absulute))
print(np.mean(relative))
plt.figure()
plt.plot(absulute)
plt.savefig('a.png')
plt.figure()
plt.plot(relative)
plt.savefig('r.png')