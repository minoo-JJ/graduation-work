# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
#from keras.layers import SimpleRNN, Conv1D, MaxPool1D, UpSampling1D
#from keras.backend import greater_equal, less

#정규화함수 선언: 처음 값 기준으로 편차로 나타내기
def normalize(_list):
    normalized = []
    for x in _list:
        normalized_x = [y/x[0] -1 for y in x]
        normalized.append(normalized_x)
    return np.array(normalized)

#하이퍼파라미터 설정하기
#추후 설정 예정

'''#상승/하강에 대한 loss function 정의 
def updown(y_true, y_pred):
    loss = 0
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    for i in range(y_true.shape[0]-1):
        if (y_pred[i]>=y_pred[i+1] and y_true[i]>=y_true[i+1]):
            loss -= 1
        elif (y_pred[i]<y_pred[i+1] and y_true[i]<y_true[i+1]): 
            loss -= 1
        else:
            loss += 1
    return loss'''
#%% pre-processing for window1
np.random.seed(10)

#2015-01-02 ~ 2021-07-03 종가 기준
_data = pd.read_csv('C:/Users/minwoo/Desktop/졸업작품/code/graduation-work/poongsan_data1.csv')
data = _data.fillna(method='ffill') #null 값을 이전 값으로 대체

whole = []

for i in range(len(data['Adj Close'].values)-60):
    whole.append(data['Adj Close'].values[i:i+61])

whole = normalize(whole)

train = whole[:1540,:]
#vld = whole[1200:1540,:]
test = whole[1540:,:]

np.random.shuffle(train)
#np.random.shuffle(vld)

x_train = train[:1200,:-1]
x_train = x_train[:, :, np.newaxis]
y_train = train[:1200,-1]

x_vld = train[1200:1540,:-1]
x_vld = x_vld[:, :, np.newaxis]
y_vld = train[1200:1540,-1]

x_test = test[:,:-1]
x_test = x_test[:, :, np.newaxis]
y_test = test[:,-1]
#%%pre-processing for window2
whole2 = []

for i in range(len(data['Adj Close'].values)-60):
    whole2.append(data['Adj Close'].values[i:i+61])

whole2 = normalize(whole2)

train2 = whole2[:1540,:]
#vld2 = whole2[1200:1540,:]
test2 = whole2[1540:,:]

np.random.shuffle(train2)
#np.random.shuffle(vld2)

x_train2 = train2[:1200,:-1]
x_train2 = x_train2[:, :, np.newaxis]
y_train2 = np.zeros(x_train2.shape[0])

for i in range(len(x_train2)):
    if x_train2[i][-2] - train2[i,-1] > 0:   # 증가 = 1
        y_train2[i] = 1
    elif x_train2[i][-1] - train2[i,-1] < 0: # 감소 = 0
        y_train2[i] = 0

x_vld2 = train2[1200:1540,:-1]
x_vld2 = x_vld2[:, :, np.newaxis]
y_vld2 = np.zeros(x_vld2.shape[0])

for i in range(len(x_vld2)):
    if x_vld2[i][-1] - train2[i,-1] > 0:
        y_vld2[i] = 1
    elif x_vld2[i][-1] - train2[i,-1] < 0:
        y_vld2[i] = 0

x_test2 = test2[:,:-1]
x_test2 = x_test2[:, :, np.newaxis]
y_test2 = np.zeros(x_test2.shape[0])

for i in range(len(x_test2)):
    if x_test2[i][-1] - test2[i,-1] > 0:
        y_test2[i] = 1
    elif x_test2[i][1] - test2[i,-1] < 0:
        y_test2[i] = 0
#%% 노이즈 추가 
noise = 0.01

x_train_noisy = train[:1200,:-1] + noise * np.random.normal(0,1,size=train[:1200,:-1].shape)
x_vld_noisy = train[1200:1540,:-1] + noise * np.random.normal(0,1,size=train[1200:1540,:-1].shape)
'''x_train_noisy2 = train2[:1250] + noise * np.random.normal(0,1,size=train2[:1250].shape)
x_vld_noisy2 = train2[1250:1600] + noise * np.random.normal(0,1,size=train2[1250:1600].shape)'''
x_train_noisy2 = train2[:1200,:-1] + noise * np.random.normal(0,1,size=train2[:1200,:-1].shape)
x_vld_noisy2 = train2[1200:1540,:-1] + noise * np.random.normal(0,1,size=train2[1200:1540,:-1].shape)
#%% 오토인코더 for window1
encoding_dim = 35
input_val = Input(shape=(60,))
encoded = Dense(encoding_dim, activation='relu')(input_val)
decoded = Dense(60, activation='linear')(encoded)

autoencoder = Model(input_val, decoded)

encoder = Model(input_val, encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

hist_auto = autoencoder.fit(x_train_noisy, train[:1200,:-1]
                            , batch_size=8, epochs=300
                            , validation_data=(x_vld_noisy, train[1200:1540,:-1])
                            , shuffle=True)
#%% 오토인코더 for window2
encoding_dim = 35
input_val = Input(shape=(60,))
encoded = Dense(encoding_dim, activation='relu')(input_val)
decoded = Dense(60, activation='sigmoid')(encoded)

autoencoder2 = Model(input_val, decoded)

encoder = Model(input_val, encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder2.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder2.compile(optimizer='adam', loss='mse')
autoencoder2.summary()

'''hist_auto2 = autoencoder2.fit(x_train_noisy2, train2[:1250]
                            , batch_size=1, epochs=10
                            , validation_data=(x_vld_noisy2, train2[1250:1600])
                            , shuffle=True)'''
hist_auto2 = autoencoder2.fit(x_train_noisy2, train2[:1200,:-1]
                            , batch_size=8, epochs=300
                            , validation_data=(x_vld_noisy2, train2[1200:1540,:-1])
                            , shuffle=True)
#%% 오토인코더 error 체크 
fig, ax = plt.subplots(2,1)

ax[0].plot(hist_auto.history['loss'], 'r', label='train loss')
ax[0].plot(hist_auto.history['val_loss'], 'b', label='validation loss')
ax[0].legend()

ax[1].plot(hist_auto2.history['loss'], 'r', label='train loss')
ax[1].plot(hist_auto2.history['val_loss'], 'b', label='validation loss')
ax[1].legend()

ax[0].set_xlabel('epochs')
ax[0].set_ylabel('loss')
ax[1].set_xlabel('epochs')
ax[1].set_ylabel('loss')

ax[0].set_ylim(0.0001,0.0003)
#ax[1].set_ylim(0,0.02)
#%% 모델1 학습
model = Sequential()
model.add(LSTM(35, input_shape=(60,1), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(70, activation='relu', return_sequences=False))
#model.add(Dropout(0.5))
#model.add(LSTM(60, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mse')
model.summary()
x_auto = autoencoder.predict(x_train_noisy)
x_auto = x_auto[:, :, np.newaxis]

hist = model.fit(x_auto, y_train, batch_size=20, epochs=80
                 , validation_data=(x_vld, y_vld))
#%% Bidirestional
x_input = Input((60,1))
x_lstm = Bidirectional(LSTM(10, return_sequences=True)\
                       , merge_mode='concat')(x_input)
x_lstm2 = Bidirectional(LSTM(20, return_sequences=False), merge_mode='concat')(x_lstm)
y_output = Dense(1, activation='linear')(x_lstm2)
model3 = Model(x_input, y_output)

model3.summary()

model3.compile(optimizer='adam', loss='mse')
x_auto = autoencoder.predict(x_train_noisy)
x_auto = x_auto[:, :, np.newaxis]

hist3 = model3.fit(x_auto, y_train, batch_size=20, epochs=80
                 , validation_data=(x_vld, y_vld))
#%% 모델2 학습
model2 = Sequential()
model2.add(LSTM(10, input_shape=(2,1), return_sequences=True))
#model.add(Dropout(0.5))
model2.add(LSTM(20, activation='relu', return_sequences=False))
#model.add(Dropout(0.5))
#model.add(LSTM(60, activation='relu'))
model2.add(Dense(2, activation='sigmoid'))

model2.compile(optimizer='adam', loss='binary_crossentropy')
model2.summary()
#x_auto2 = autoencoder2.predict(x_train_noisy2)
#x_auto2 = x_auto2[:, :, np.newaxis]

hist2 = model2.fit(x_train2, y_train2, batch_size=2, epochs=80
                 , validation_data=(x_vld2, y_vld2))
#%% Bidirectional
x_input = Input((60,1))
x_lstm = LSTM(60, return_sequences=True)(x_input)
x_lstm2 = LSTM(120)(x_lstm)
y_output = Dense(1, activation='sigmoid')(x_lstm2)
model4 = Model(x_input, y_output)

model4.summary()

model4.compile(optimizer='adam', loss='binary_crossentropy')
#x_auto2 = autoencoder2.predict(x_train_noisy2)
#x_auto2 = x_auto2[:, :, np.newaxis]

hist4 = model4.fit(x_train2, y_train2, batch_size=2, epochs=50
                 , validation_data=(x_vld2, y_vld2))
#%% 모델 error 체크 
fig, ax = plt.subplots(2,1)

ax[0].plot(hist3.history['loss'], 'r', label='train loss')
ax[0].plot(hist3.history['val_loss'], 'b', label='validation loss')
ax[0].legend()

ax[1].plot(hist4.history['loss'], 'r', label='train loss')
ax[1].plot(hist4.history['val_loss'], 'b', label='validation loss')
ax[1].legend()

ax[0].set_xlabel('epochs')
ax[0].set_ylabel('loss')
ax[1].set_xlabel('epochs')
ax[1].set_ylabel('loss')

ax[0].set_ylim(0,0.002)
#ax[1].set_ylim(0,0.03)
#%% 예측 결과
y_hat = model3.predict(x_test)
y_hat2 = model4.predict(x_test2)

fig, ax = plt.subplots(2,1)

ax[0].plot(y_hat, 'r', label = "predicted", linewidth=1)
ax[0].plot(y_test, 'black', label = "actual", linewidth=1)
ax[0].legend()

ax[1].plot(y_hat2, 'r', label = "predicted", linewidth=1)
ax[1].plot(y_test2, 'black', label = "actual", linewidth=1)
ax[1].legend()

ax[0].set_xlabel('days')
ax[0].set_ylabel('value')
ax[1].set_xlabel('days')
ax[1].set_ylabel('value')

#plt.ylim(0,0.05)