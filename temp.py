# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, SimpleRNN, Dropout

#정규화함수 선언: 처음 값 기준, 편차로 나타내기
def normalize(_list):
    normalized = []
    for x in _list:
        normalized_x = [y/x[0] -1 for y in x]
        normalized.append(normalized_x)
    return np.array(normalized)

#하이퍼파라미터 설정하기


np.random.seed(100)

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

model = Sequential()
model.add(LSTM(100, input_shape=(60,1)))
#model.add(Dropout(0.5))
model.add(Dense(120, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.summary()
hist = model.fit(x_train, y_train, batch_size=10, epochs=100, validation_data=(x_vld, y_vld))
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

y_hat = model.predict(x_test)

plt.figure()
plt.plot(y_hat, 'r', label = "predicted", linewidth=1)
plt.plot(y_test, 'black', label = "actual", linewidth=1)
plt.legend()

plt.xlabel('days')
plt.ylabel('value')