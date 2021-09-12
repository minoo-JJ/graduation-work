# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, SimpleRNN
from keras.layers import Input, Conv1D, MaxPool1D, UpSampling1D

#정규화함수 선언: 처음 값 기준, 편차로 나타내기
def normalize(_list):
    normalized = []
    for x in _list:
        normalized_x = [y/x[0] -1 for y in x]
        normalized.append(normalized_x)
    return np.array(normalized)

#하이퍼파라미터 설정하기


np.random.seed(10)

#2015-01-02 ~ 2021-07-03 종가 기준
_data = pd.read_csv('C:/Users/minwoo/Desktop/졸업작품/code/graduation-work/poongsan_data1.csv')
data = _data.fillna(method='ffill') #null 값을 이전 값으로 대체

whole = []
for i in range(len(data['Adj Close'].values)-60):
    whole.append(data['Adj Close'].values[i:i+61])

whole = normalize(whole)

train = whole[:1200,:]
vld = whole[1200:1540,:]
test = whole[1540:,:]

np.random.shuffle(train)
np.random.shuffle(vld)

x_train = train[:,:-1]
x_train = x_train[:, :, np.newaxis]
y_train = train[:,-1]

x_vld = vld[:,:-1]
x_vld = x_vld[:, :, np.newaxis]
y_vld = vld[:,-1]

x_test = test[:,:-1]
x_test = x_test[:, :, np.newaxis]
y_test = test[:,-1]
#%% 노이즈 추가 

noise = 0.01

x_train_noisy = x_train + noise * np.random.normal(0,1,size=x_train.shape)
x_vld_noisy = x_vld + noise * np.random.normal(0,1,size=x_vld.shape)

#%%
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

hist_auto = autoencoder.fit(x_train_noisy, x_train
                            , batch_size=8, epochs=200
                            , validation_data=(x_vld_noisy, x_vld)
                            , shuffle=True)
#%%  오류 해결 필요 
'''input_val = Input(shape=(60,))

x = Conv1D(35, 10, activation='relu', padding='same')(input_val)
x = MaxPool1D(5, padding='same')(x)
x = Conv1D(35, 10, activation='relu', padding='same')(x)
encoded = MaxPool1D(5, padding='same')(x)

x = Conv1D(35, 10, activation='relu', padding='same')(encoded)
x = UpSampling1D(5)(x)
x = Conv1D(35, 10, activation='relu', padding='same')(x)
x = UpSampling1D(5)(x)
decoded = Conv1D(1, 10, activation='linear', padding='same')(x)

autoencoder = Model(input_val, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()'''

Model = Sequential()
Model.add(Conv1D(35, 10, input_shape=(60,1), activation='relu', padding='same'))
Model.add(MaxPool1D(5, padding='same'))
Model.add(Conv1D(35, 10, activation='relu', padding='same'))
Model.add(MaxPool1D(5, padding='same'))
Model.add(Conv1D(35, 10, activation='relu', padding='same'))
Model.add(UpSampling1D(5))
Model.add(Conv1D(35, 10, activation='relu', padding='same'))
Model.add(UpSampling1D(5))
Model.add(Dense(60, activation='linear'))

hist_auto = autoencoder.fit(x_train_noisy, x_train
                            , batch_size=8, epochs=200
                            , validation_data=(x_vld_noisy, x_vld)
                            , shuffle=True)
#%%
plt.figure()
plt.plot(hist_auto.history['loss'], 'r', label='train loss')
plt.plot(hist_auto.history['val_loss'], 'b', label='validation loss')
plt.legend()

plt.xlabel('epochs')
plt.ylabel('loss')

#plt.ylim(0,0.001)
#%%
model = Sequential()
model.add(LSTM(35, input_shape=(60,1), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(70, activation='relu', return_sequences=False))
#model.add(Dropout(0.5))
#model.add(LSTM(60, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

x_auto = autoencoder.predict(x_train_noisy)
x_auto = x_auto[:, :, np.newaxis]

hist = model.fit(x_auto, y_train, batch_size=20, epochs=50
                 , validation_data=(x_vld, y_vld))
#%%
fig, ax = plt.subplots(2,1)

ax[0].plot(hist.history['loss'], 'r', label='train loss')
ax[0].plot(hist.history['val_loss'], 'b', label='validation loss')
ax[0].legend()

ax[1].plot(hist.history['mae'], 'r', label='train loss')
ax[1].plot(hist.history['val_mae'], 'b', label='validation loss')
ax[1].legend()

ax[0].set_xlabel('epochs')
ax[0].set_ylabel('loss')
ax[1].set_xlabel('epochs')
ax[1].set_ylabel('mae')

ax[0].set_ylim(0,0.004)
ax[1].set_ylim(0,0.05)

y_hat = model.predict(x_test)

plt.figure()
plt.plot(y_hat, 'r', label = "predicted", linewidth=1)
plt.plot(y_test, 'black', label = "actual", linewidth=1)
plt.legend()

plt.xlabel('days')
plt.ylabel('value')