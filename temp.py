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

#하이퍼파라미터 설정


np.random.seed(100)

_data = pd.read_csv('poongsan_data.csv') #2015-01-02 ~ 2021-07-02 종가 기준
data = _data.fillna(method='ffill') #null 값을 이전 값으로 대체

whole = []
for i in range(len(data['Adj Close'].values)-60):
    whole.append(data['Adj Close'].values[i:i+61])

whole = normalize(whole)

train = whole[:1200,:]
vld = whole[1200:,:]

np.random.shuffle(train)

x_train = train[:,:-1]
x_train = x_train[:, :, np.newaxis]
y_train = train[:,-1]

x_vld = vld[:,:-1]
x_vld = x_vld[:, :, np.newaxis]
y_vld = vld[:,-1]

'''
x_trained = []
x_test = []
 
y_trained = data['Adj Close'].values[60:1414+60]
#y_trained = pd.DataFrame(data=y_trained)

y_test = data['Adj Close'].values[1414+60:1599]
#y_test = pd.DataFrame(data=y_test)

for i in range(len(y_trained)):
    x_trained.append(data['Adj Close'].values[i:i+60])    
#df_trained = pd.DataFrame(data = np.c_[x_trained, y_trained], columns = np.linspace(-60, 0, 61))

x_trained = normalize(x_trained)
x_trained = x_trained[:, :, np.newaxis]
#print(x_trained.shape, y_trained.shape)

for i in range(len(y_test)):
    x_test.append(data['Adj Close'].values[1474-60+i:1474+i])
#df_test = pd.DataFrame(data = np.c_[x_test, y_test], columns = np.linspace(-60, -1, 61))

x_test = normalize(x_test)
x_test = x_test[:, :, np.newaxis]
#print(x_test.shape, y_test.shape)

plt.subplot(2,1,1)
plt.plot(y_trained, linewidth=0.5)
plt.subplot(2,1,2)
plt.plot(y_test, linewidth=1)
#plt.plot(y_test, 'b', linewidth=1)
'''

model = Sequential()
model.add(LSTM(60, input_shape=(60,1)))
#model.add(Dropout(0.5))
model.add(Dense(80, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.summary()
hist = model.fit(x_train, y_train, batch_size=10, epochs=50, validation_data=(x_vld, y_vld))

#%%
_test = pd.read_csv('poongsan_test.csv') #2021-07-03 ~ 2021-09-02 종가 기준
test = _test.fillna(method='ffill')

x_test = []
x_test.append(data['Adj Close'].values[-60:-1])

fig, ax = plt.subplots(2,1)

ax[0].plot(hist.history['loss'], 'r', label='train loss')
ax[0].plot(hist.history['val_loss'], 'b', label='validation loss')
ax[0].legend()
'''
ax[1].plot(hist.history['mae'], 'r', label='train loss')
ax[1].plot(hist.history['val_mae'], 'b', label='validation loss')
ax[1].legend()
'''
ax[0].set_xlabel('epochs')
ax[0].set_ylabel('loss')
'''ax[1].set_xlabel('epochs')
ax[1].set_ylabel('mae')'''

x_hat = x_vld
x_hat = np.array(x_hat)
y_hat = model.predict(x_hat)
y_hat = np.array(y_hat)

ax[1].plot(y_hat, 'r', label = "predicted", linewidth=1)
ax[1].plot(y_vld, 'black', label = "actual", linewidth=1)
ax[1].legend()

ax[1].set_xlabel('days')
ax[1].set_ylabel('value')