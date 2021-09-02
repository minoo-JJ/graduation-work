# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, SimpleRNN

#정규화함수 선언

#하이퍼파라미터 설정
def sigmoid(self, x):
    return 1.0/(1.0 + np.exp(-x))

def normalize(self, x):
    return (x / 255.0) * 0.99 + 0.01
    
def tanh(self, x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
        
def softmax(self, x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


np.random.seed(100)
data = pd.read_csv('poongsan_data.csv')

x_trained = []
x_test = []

#2015-01-02 ~ 2021-07-02 종가 기준 
y_trained = data['Adj Close'].values[60:1474]
#y_trained = pd.DataFrame(data=y_trained)

y_test = data['Adj Close'].values[1474:1599]
#y_test = pd.DataFrame(data=y_test)

for i in range(len(y_trained)):
    x_trained.append(data['Adj Close'].values[i:i+60])
#df_trained = pd.DataFrame(data = np.c_[x_trained, y_trained], columns = np.linspace(-60, 0, 61))
x_trained=np.array(x_trained)
x_trained = x_trained[:, :, np.newaxis]
#print(x_trained.shape, y_trained.shape)

for i in range(len(y_test)):
    x_test.append(data['Adj Close'].values[1474-60+i:1474+i])
#df_test = pd.DataFrame(data = np.c_[x_test, y_test], columns = np.linspace(-60, -1, 61))
x_test=np.array(x_test)
x_test = x_test[:, :, np.newaxis]
#print(x_test.shape, y_test.shape)


plt.subplot(2,1,1)
plt.plot(y_trained, linewidth=0.5)
plt.subplot(2,1,2)
plt.plot(y_test, linewidth=1)
#plt.plot(y_test, 'b', linewidth=1)


model = Sequential()
model.add(SimpleRNN(60, input_shape=(60,1)))
model.add(Dense(60, activation='relu'))
model.add(Dense(1, activation='softmax'))

model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])

model.fit(x_trained, y_trained, epochs=20)
'''
0-1473
1473-60

predict = model.predict([[750, 3.7, 3], [400, 2.2, 1]])
print(predict)
'''
#%%
predict = model.predict(x_test)
print(predict)