# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 19:38:18 2023

@author: 11527
"""
import keras
from keras import backend as K
from keras.layers import (Activation, AveragePooling1D, BatchNormalization,
                          Conv1D, Dense, GlobalAveragePooling1D, Input,Dropout)
from tensorflow.python.keras.layers.core import Lambda


from keras.models import Model
from keras.optimizers import Nadam
from keras.regularizers import l2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.layers import Reshape
from keras.models import load_model

def abs_backend(inputs):
    return K.abs(inputs)


def expand_dim_backend(inputs):
    return K.expand_dims(inputs, 1)


def sign_backend(inputs):
    return K.sign(inputs)


def pad_backend(inputs, in_channels, out_channels):
    pad_dim = (out_channels - in_channels) // 2
    inputs = K.expand_dims(inputs)
    inputs = K.spatial_2d_padding(inputs, padding=((0, 0), (pad_dim, pad_dim)))
    return K.squeeze(inputs, -1)


def residual_shrinkage_block(incoming, nb_blocks, out_channels, downsample=False,
                             downsample_strides=2):
    """A residual_shrinkage_block.

    Arguments:
        incoming: input tensor.
        nb_blocks: integer, numbers of block.
        out_channels: integer, filters of the conv1d layer.
        downsample: default False, downsample or not.
        downsample_strides: default 2, stride of the first layer.

    Returns:
        Output tensor for the residual block.
    """

    residual = incoming
    in_channels = incoming.get_shape().as_list()[-1]

    for _ in range(nb_blocks):

        identity = residual

        if not downsample:
            downsample_strides = 1

        residual = BatchNormalization()(residual)
        residual = Activation('relu')(residual)
        residual = Conv1D(out_channels, 3, strides=downsample_strides,
                          padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=l2(1e-4))(residual)

        residual = BatchNormalization()(residual)
        residual = Activation('relu')(residual)
        residual = Conv1D(out_channels, 3, padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=l2(1e-4))(residual)

        # Calculate global means
        residual_abs = Lambda(abs_backend)(residual)
        abs_mean = GlobalAveragePooling1D()(residual_abs)

        # Calculate scaling coefficients
        scales = Dense(out_channels, activation=None, kernel_initializer='he_normal',
                       kernel_regularizer=l2(1e-4))(abs_mean)
        scales = BatchNormalization()(scales)
        scales = Activation('relu')(scales)
        scales = Dense(out_channels, activation='sigmoid', kernel_regularizer=l2(1e-4))(scales)
        scales = Lambda(expand_dim_backend)(scales)

        # Calculate thresholds
        thres = keras.layers.multiply([abs_mean, scales])

        # Soft thresholding
        sub = keras.layers.subtract([residual_abs, thres])
        zeros = keras.layers.subtract([sub, sub])
        n_sub = keras.layers.maximum([sub, zeros])
        residual = keras.layers.multiply([Lambda(sign_backend)(residual), n_sub])

        # Downsampling using the pooL-size of (1, 1)
        if downsample_strides > 1:
            identity = AveragePooling1D(pool_size=1, strides=2)(identity)

        # Zero_padding or Conv1D to match channels
        if in_channels != out_channels:
            """ padding """
            identity = Lambda(pad_backend, arguments={'in_channels': in_channels, 'out_channels': out_channels})(
                identity)
            """ Conv1D """
            # identity = Conv1D(out_channels,1,strides=1,padding='same')(identity)

        residual = keras.layers.add([residual, identity])

    return residual


if __name__ == '__main__':
    inputs = 1116
    outputs = 1116

    x_input = Input(shape=(inputs, 1))
    x = Conv1D(8, 3, 2, padding='same')(x_input)
    x = residual_shrinkage_block(x, 1, 16, downsample=True)
    x = residual_shrinkage_block(x, 3, 16, downsample=False)

    x = residual_shrinkage_block(x, 1, 32, downsample=True)
    x = residual_shrinkage_block(x, 3, 32, downsample=False)

    x = residual_shrinkage_block(x, 1, 64, downsample=True)
    x = residual_shrinkage_block(x, 3, 64, downsample=False)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #shape = x.get_shape().as_list()
    #x = Reshape((shape[1] * shape[2],))(x)
    x = GlobalAveragePooling1D()(x)
    print(x.shape)

    x = Dense(128, activation='sigmoid')(x)
    x = Dense(512, activation='sigmoid')(x)
    x = Dense(outputs, activation='sigmoid')(x)
    model = Model(inputs=x_input, outputs=x)
    model.load_weights('exp.h5')

   # for layer in model.layers:
   #    layer.trainable = False
    # 获取模型的输出
    previous_output = model.output
    x = Dense(1,activation='sigmoid')(previous_output)
    new_model = Model(inputs=model.input,outputs=x)
    #new_model.load_weights('my_model_re.h5')

    optimizers = Nadam(learning_rate=1e-5)
    model.compile(optimizer=optimizers, loss='mean_squared_error')
    new_model.compile(optimizer=optimizers, loss='mean_squared_error')



    #model.summary()
    train_data = np.load('F:\Wujiahui\deoniseProject\\train_data_CH4.npy')
    test_data = np.load('F:\Wujiahui\deoniseProject\\test_data_CH4.npy')
    test_concertration = np.linspace(10,1000,10000)/1000
    plt.figure()
    plt.plot(test_data[1])
    plt.plot(test_data[10])
    plt.plot(test_data[100])
    plt.show()
    # test_concertration = np.load("F:\Wujiahui\deoniseProject\\test_data_CH4.npy")
    x_train_1, x_test_1, y_train_1, y_test_1, y_train_2, y_test_2 = train_test_split(train_data, test_data,test_concertration, test_size=0.1,random_state=2)
    np.save('F:\Wujiahui\deoniseProject\git_x.npy', x_test_1)
    np.save('F:\Wujiahui\deoniseProject\git_y_A.npy', y_test_1)
    np.save('F:\Wujiahui\deoniseProject\git_y_B.npy', y_test_2)


    history = model.fit(x_train_1,y_train_1,batch_size=32,epochs=2,validation_data=(x_test_1,y_test_1))
    model.save('exp.h5')
    #new_model.load_weights('my_model_re.h5')
    #history = new_model.fit(x_train_1,y_train_2,batch_size=16,epochs=100,validation_data=(x_test_1,y_test_2))
    #new_model.save('my_model_re.h5')
    #score = model.evaluate(x_test_1,y_test_1)
    ''' 
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    new_model.load_weights('exp.h5')
    prediction_2 = new_model.predict(x_test_1)
    np.save('Pre_DNSR',prediction_2)
    '''
    model.load_weights('exp.h5')
    prediction= model.predict(x_test_1)
    np.save('Pre_DNSR',prediction)
















