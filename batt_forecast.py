#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 17:10:02 2018

@author: scsingh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, LeakyReLU, GRU, SimpleRNN

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import batt_caiso

def scale(P, scaler):
    Pnorm = scaler.fit_transform(P.reshape(-1, 1))
    return Pnorm

# convert an array of values into a dataset matrix
def create_dataset(dataset, out_size, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        b = dataset[(i+look_back):(i+look_back+out_size)]
        if len(b) < out_size:
            pad = np.zeros(out_size - len(b)).reshape(-1, 1)
            b = np.append(b, pad).reshape(-1, 1)
        dataY.append(b)
    return np.array(dataX), np.array(dataY)

def caiso_fcst(P, typez, tr_size=552, lag = 5, out_size=1):
        
    scaler = MinMaxScaler(feature_range=(0, 1))
    Pnorm = scale(P, scaler)
    plt.plot(Pnorm)
    
    #ind = np.arange(len(Pnorm))
    #X, Y, indX, indY = create_dataset(Pnorm, ind)
    
    look_back = lag*out_size
    trainX, trainY = create_dataset(Pnorm[:tr_size], out_size, look_back)
    testX, testY = create_dataset(Pnorm[tr_size:], out_size, look_back)
    
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    
    trainY = np.reshape(trainY, (trainY.shape[0], trainY.shape[1]))
    testY = np.reshape(testY, (testY.shape[0], testY.shape[1]))
     
    print ('\n\n Running Model type: {}'.format(typez))
    model = Sequential()
    if typez=='VRNN':
        model.add(SimpleRNN(units = 500, activation = 'tanh', input_shape=(1, look_back), return_sequences=False))
    elif typez=='LSTM':
        model.add(LSTM(units = 500, activation = 'tanh', input_shape=(1, look_back), return_sequences=False))
    elif typez=='GRU':
        model.add(GRU(units = 500, activation = 'tanh', input_shape=(1, look_back), return_sequences=False))
    else:
        print('wrong option')
    model.add(Dropout(0.8))
    model.add(Dense(out_size)) 
    model.add(LeakyReLU())
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, epochs=20, batch_size=10, validation_data=(testX, testY), verbose=1)
    #model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
    #model.save('./savedModel')
    
    #predict
    trainHat = model.predict(trainX)
    testHat = model.predict(testX)
    
    #rmse
    trainScore = math.sqrt(mean_squared_error(trainY, trainHat))
    print('Train: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY, testHat))
    print('Test: %.2f RMSE' % (testScore))
    
    #invert 
    trainHat_inv = scaler.inverse_transform(trainHat)
    trainY_inv = scaler.inverse_transform(trainY)
    testHat_inv = scaler.inverse_transform(testHat)       
    testY_inv = scaler.inverse_transform(testY)
    
    #generate plots
    trainHat_inv_sub = trainHat_inv[1::24]
    trainY_inv_sub = 
    testHat_inv_sub = 
    testY_inv_sub = 
    #shift train predictions for plotting
    trainPredictPlot = np.empty_like(Pnorm)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainHat)+look_back, :] = trainHat
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(Pnorm)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainHat)+(look_back*2)+1:len(Pnorm)-1, :] = testHat
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(Pnorm))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()

P = batt_caiso.caiso_price('CAISO_201809.csv', show_info=False, show_plot=True)
optionz = ['VRNN', 'LSTM', 'GRU']
modelz = optionz[1]
tr_size = 552
out_size = 24
lag = 5
caiso_fcst(P, modelz, tr_size, lag, out_size)
