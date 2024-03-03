import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
import pandas as pd
# -*- coding: utf-8 -*-
from hapi import *
from numpy import arange
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from pylab import show,plot,subplot,xlim,ylim,title,legend,xlabel,ylabel
import os
matplotlib.use('Qt5Agg')

db_begin('CH4')
fetch('CH4',6,1,6045.7,6048)
tableList()
describeTable('CH4')

#绘制线强图
x,y = getStickXY('CH4')

plt.figure()

plot(x,y)

xlabel('wavenumber($cm{-1}$)')
ylabel('$HCH4 linestrength  ')
# 用voigt线性计算吸收系数Kv
# 用voigt线性计算吸收系数Kv
concentration = 0.01/100 # 生成100ppm的CH4吸收光谱
Nu,Coef = absorptionCoefficient_Voigt(((6,1),),'CH4',WavenumberStep=0.002,Environment={'p':1,'T':296},
OmegaStep = 0.1,HITRAN_units=False,GammaL='gamma_self',Diluent={'air':1-concentration,'self':concentration})
Coef *= concentration
#计算吸光度光谱
Absorbance = Coef * 4000
Nu , tras_hitran = transmittanceSpectrum(Nu,Coef,Environment={'l':4000})
#将横坐标转换为频率
plt.figure()
plt.plot(Nu,Absorbance,color='red')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
f_Nu = (1/Nu)*1e7
#计算投射光谱
#tras_hitran = tras_hitran[::-1]
plt.figure()
plt.plot(Nu,tras_hitran)
plt.xlabel('wavenumber($cm^{-1}$)')
plt.ylabel('trasition')
plt.title('$ CH42 absorption spectra @ 1atm, 296K,L=4000cm $')
path=os.getcwd()


def wgn(x, snr):
    Ps = np.sum(abs(x)**2)/len(x)
    Pn = Ps/(10**((snr/10)))
    noise = np.random.randn(len(x)) * np.sqrt(Pn)
    signal_add_noise = x + noise
    return signal_add_noise

def noisy_spectrum(label):
    '''
    label : 给定的确定浓度的吸光度曲线
    snr: 指定信号噪声的大小
    '''
    length = len(label)
    label_spectrum = np.tile(label,(1000,1))
    Noisy_spectrum = np.zeros([1000,length])
    factors =np.linspace(0.01, 1, 1000)
    label_spectrum = label_spectrum * factors[:,np.newaxis]
    label_spectrum = exp(-label_spectrum)

    for index,i in enumerate(label_spectrum):
        Noisy_spectrum[index:] = i + np.random.randn(length) * 0.00686
    return label_spectrum,Noisy_spectrum

def base_line(signal):
    x = np.arange(signal.shape[1])
    baseline_signal = np.zeros([signal.shape[0],signal.shape[1]])
    coefficients = [8.10071477e-12, - 2.49387737e-08 , 3.85763541e-04 , 2.60286352e+00]

    for i,spectrum in enumerate(signal):
        y_fit = np.polyval(coefficients, x)
        baseline_signal[i:] = y_fit * spectrum
    return baseline_signal


CH4_label,CH4_10dB = noisy_spectrum(Absorbance)
baseline_noisyspectrum =  base_line(CH4_10dB)
baseline_spectrum =  base_line(CH4_label)

plt.figure()
plt.plot(CH4_10dB[5])
plt.plot(CH4_10dB[55])
plt.plot(CH4_10dB[555])
plt.plot(CH4_10dB[999])

plt.show()

plt.figure()
plt.plot(baseline_noisyspectrum[55])

plt.plot(baseline_spectrum[55])
plt.show()
print(1)
np.save('CH4_train.npy',CH4_10dB)
np.save('CH4_test.npy',CH4_label)



a = np.load('F:\Wujiahui\gas_retrieval_with_deep_learning-main\Simulated-DAS-data\CH4_20dB.pkl.npy')
b = np.load('F:\Wujiahui\gas_retrieval_with_deep_learning-main\Simulated-DAS-data\CH4_0dB.pkl.npy')

plt.figure()
plt.plot(a[9999])
plt.plot(b[9999])


def calculate_snr(signal, noisy_signal):
    # 计算信号的均方根（RMS）
    signal_rms = np.sqrt(np.mean(np.square(signal)))
    # 计算噪声的均方根（RMS）
    noise = noisy_signal - signal
    noise_rms = np.sqrt(np.mean(np.square(noise)))
    # 计算信噪比（SNR）
    snr = 10 * np.log10(signal_rms / noise_rms)
    return snr
plt.show()
plt.figure()
plt.plot(a[9999]-b[9999])
plt.show()
print(calculate_snr(b[9999],a[9999]))
print(calculate_snr(baseline_spectrum[666],baseline_noisyspectrum[666]))