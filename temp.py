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

#정규화함수 선언
def normalize(_list):
    normalized = []
    for x in _list:
        normalized_x = [y/x[0] -1 for y in x]
        normalized.append(normalized_x)
    return np.array(normalized)

#하이퍼파라미터 설정


np.random.seed(100)
data = pd.read_csv('poongsan_data.csv')

whole = []
for i in range(len(data['Adj Close'].values)-61):
    whole.append(data['Adj Close'].values[i:i+61])

whole = normalize(whole)

train = whole[:1200,:]
vld = whole[1200:,:]

x_train = train[:,:-1]
x_train = x_train[:, :, np.newaxis]
y_train = train[:,-1]

x_vld = vld[:,:-1]
x_vld = x_vld[:, :, np.newaxis]
y_vld = vld[:,-1]

x_trained = []
x_test = []

###여기까지 60개 이전 종가 + 1개 예측 값을 한 벡터로 합쳐 놓음


#2015-01-02 ~ 2021-07-02 종가 기준 
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


model = Sequential()
model.add(LSTM(100, input_shape=(60,1)))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='softmax'))

model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])

model.summary()
model.fit(x_trained, y_trained, epochs=20)

#%%
predict = model.predict(x_test)
print(predict)