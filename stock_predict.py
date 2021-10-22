# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, Input, Bidirectional

# Data normalization
def normalize(_list):
    normalized = []
    for x in _list:
        normalized_x = [y/x[0] -1 for y in x]
        normalized.append(normalized_x)
    return np.array(normalized)
#%% Load data
#2008-07-30 ~ 2021-10-20 종가 기준
_data = pd.read_csv('C:/Users/minwoo/Desktop/졸업작품/code/graduation-work/data.csv')
data = _data.fillna(method='ffill') #null 값을 이전 값으로 대체
#%% Parameter setting
window_size = 60
train_size = int(len(data) * 0.8)
valid_size = int(train_size * 0.9)
#%% Pre-processing for model1 
np.random.seed(10)

whole = []

for i in range(len(data['Adj Close'].values) - window_size):
    whole.append(data['Adj Close'].values[i:i+(window_size+1)])

whole = normalize(whole)

train = whole[:train_size,:]
test = whole[train_size:,:]

np.random.shuffle(train)

x_train = train[:valid_size,:-1]
x_train = x_train[:, :, np.newaxis]
y_train = train[:valid_size,-1]

x_vld = train[valid_size:train_size,:-1]
x_vld = x_vld[:, :, np.newaxis]
y_vld = train[valid_size:train_size,-1]

x_test = test[:,:-1]
x_test = x_test[:, :, np.newaxis]
y_test = test[:,-1]
#%% Pre-processing for window2
train2 = whole[:train_size,:]
test2 = whole[train_size:,:]

np.random.shuffle(train2)

x_train2 = train2[:train_size,:-1] # valid->train 수정함
x_train2 = x_train2[:, :, np.newaxis]
y_train2 = np.zeros(x_train2.shape[0])

for i in range(len(x_train2)):
    if x_train2[i][-1] - train2[i,-1] > 0:   # 감소 = 0
        y_train2[i] = 0
    elif x_train2[i][-1] - train2[i,-1] < 0: # 증가 = 1
        y_train2[i] = 1

# x_vld2 = train2[valid_size:train_size,:-1]
# x_vld2 = x_vld2[:, :, np.newaxis]
# y_vld2 = np.zeros(x_vld2.shape[0])

# for i in range(len(x_vld2)):
#     if x_vld2[i][-1] - train2[valid_size+i,-1] > 0:
#         y_vld2[i] = 0
#     elif x_vld2[i][-1] - train2[valid_size+i,-1] < 0:
#         y_vld2[i] = 1

x_test2 = test2[:,:-1]
x_test2 = x_test2[:, :, np.newaxis]
y_test2 = np.zeros(x_test2.shape[0])
# y_test2 = test2[:,-1]

for i in range(len(x_test2)):
    if x_test2[i][-1] - test2[i,-1] > 0:
        y_test2[i] = 0
    elif x_test2[i][-1] - test2[i,-1] < 0:
        y_test2[i] = 1
#%% Noise  
noise = 0.01

x_train_noisy = train[:valid_size,:-1] + noise * np.random.normal(0,1,size=train[:valid_size,:-1].shape)
x_vld_noisy = train[valid_size:train_size,:-1] + noise * np.random.normal(0,1,size=train[valid_size:train_size,:-1].shape)
x_train_noisy2 = train2[:valid_size,:-1] + noise * np.random.normal(0,1,size=train2[:valid_size,:-1].shape)
x_vld_noisy2 = train2[valid_size:train_size,:-1] + noise * np.random.normal(0,1,size=train2[valid_size:train_size,:-1].shape)
#%% Autoencoder for model1
encoding_dim = 35
input_val = Input(shape=(window_size,))
encoded = Dense(encoding_dim)(input_val)
decoded = Dense(window_size, activation='linear')(encoded)

autoencoder = Model(input_val, decoded)

encoder = Model(input_val, encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

hist_auto = autoencoder.fit(x_train_noisy, train[:valid_size,:-1]
                            , batch_size=8, epochs=100
                            , validation_data=(x_vld_noisy, train[valid_size:train_size,:-1])
                            , shuffle=True)
#%% Autoencoder for train2
encoding_dim = 35
input_val = Input(shape=(window_size,))
encoded = Dense(encoding_dim)(input_val)
decoded = Dense(window_size, activation='linear')(encoded)

autoencoder2 = Model(input_val, decoded)

encoder = Model(input_val, encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder2.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder2.compile(optimizer='adam', loss='mse')
autoencoder2.summary()

hist_auto2 = autoencoder2.fit(x_train_noisy2, train2[:valid_size,:-1]
                            , batch_size=8, epochs=100
                            , validation_data=(x_vld_noisy2, train2[valid_size:train_size,:-1])
                            , shuffle=True)
#%% Autoencoder error plotting 
fig, ax = plt.subplots(2,1,figsize=(8,8))

ax[0].plot(hist_auto.history['loss'], 'r', label='train loss')
ax[0].plot(hist_auto.history['val_loss'], 'b', label='validation loss')
ax[0].legend()

ax[1].plot(hist_auto2.history['loss'], 'r', label='train loss')
ax[1].plot(hist_auto2.history['val_loss'], 'b', label='validation loss')
ax[1].legend()

ax[0].set_xlabel('epochs')
ax[0].set_ylabel('loss')
ax[1].set_ylabel('loss')

ax[0].set_ylim(0.0001,0.0005)
ax[1].set_ylim(0.0001,0.0005)
#%% Model1 training
model1 = Sequential()
model1.add(Bidirectional(LSTM(20, input_shape=(window_size,1), return_sequences=True), merge_mode='concat'))
model1.add(Bidirectional(LSTM(20, return_sequences=False), merge_mode='concat'))
#model1.add(Bidirectional(LSTM(40, return_sequences=True), merge_mode='concat'))
#model1.add(Bidirectional(LSTM(20, return_sequences=False), merge_mode='concat'))
model1.add(Dense(1, activation='linear'))

model1.compile(optimizer='adam', loss='mse')

hist1 = model1.fit(x_train, y_train, batch_size=64, epochs=200, validation_data=(x_vld, y_vld))
model1.summary()
#%% Model2 training
from sklearn.model_selection import StratifiedKFold

model2 = Sequential()
model2.add(Bidirectional(LSTM(20, input_shape=(window_size,1), return_sequences=True), merge_mode='concat'))
model2.add(Bidirectional(LSTM(40, return_sequences=False), merge_mode='concat'))
model2.add(Dense(1, activation='sigmoid'))

# optim = tf.optimizers.RMSprop(learning_rate=0.01, rho=0.7, momentum=0.9, epsilon=1e-07)
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

n_fold = 40 
seed = 42

cv = StratifiedKFold(n_splits = n_fold, shuffle=True, random_state=seed)

y_hat3 = np.zeros((x_test2.shape[0], 1))

for i, (i_trn, i_val) in enumerate(cv.split(x_train2, y_train2), 1):
    print(f'training model for CV #{i}')
    hist2 = model2.fit(x_train2[i_trn], y_train2[i_trn], validation_data=(x_train2[i_val], y_train2[i_val]), epochs=20, batch_size=32)                 
    y_hat3 += model2.predict(x_test2) / n_fold  
#%% Model error plotting
fig, ax = plt.subplots(2,1,figsize=(8,8))

ax[0].plot(hist1.history['loss'], 'r', label='train loss')
ax[0].plot(hist1.history['val_loss'], 'b', label='validation loss')
ax[0].legend()

ax[1].plot(hist2.history['accuracy'], 'r', label='train accuracy')
ax[1].plot(hist2.history['val_accuracy'], 'b', label='validation accuracy')
ax[1].legend()

ax[0].set_ylabel('loss')
ax[1].set_xlabel('epochs')
ax[1].set_ylabel('accuracy')

ax[0].set_ylim(0,0.002)
#ax[1].set_ylim(0,0.75)
#%% Prediction ploting
import matplotlib.ticker as ticker

y_hat = model1.predict(x_test)
y_hat2 = model2.predict(x_test2)

for i in range(len(y_hat3)):
  y_hat3[i] = 1 if y_hat3[i] >= 0.5 else 0

fig, ax = plt.subplots(2,1,figsize=(10,10))
# ax[0].grid(True)
# ax[1].grid(True)
# ax[0].xaxis.set_major_locator(ticker.MultipleLocator(1))
# ax[1].xaxis.set_major_locator(ticker.MultipleLocator(1))
# ax[1].yaxis.set_major_locator(ticker.MultipleLocator(0.1))

ax[0].plot(y_hat, 'r', label = "predicted", linewidth=1)
ax[0].plot(y_test, 'black', label = "actual", linewidth=1)
ax[0].legend()

ax[1].plot(y_hat3, 'r', label = "predicted", linewidth=1)
ax[1].plot(y_test2, 'black', label = "actual", linewidth=1)
ax[1].legend()

ax[0].set_ylabel('value')
ax[1].set_ylabel('value')
ax[1].set_xlabel('days')

# ax[0].set_xlim(150,200)
ax[1].set_xlim(190,240)