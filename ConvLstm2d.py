# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 00:27:39 2021

@author: MANOJ KUMAR
"""

import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from matplotlib import pyplot
import pandas as pd
import numpy as np
import keras
from keras.layers import Input, Dense, Dropout, Concatenate,Flatten, BatchNormalization
from keras.layers import Conv1D, Conv2D, MaxPooling2D,MaxPooling3D, ConvLSTM2D, Conv3D,LSTM, AveragePooling2D, AveragePooling3D
from keras.models import Sequential, Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from keras import regularizers
import math
import time
# load the new file
spe = pd.read_csv('speeddatactrg2021.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
print(spe.head())
print(spe.shape)
# save
#spe.to_csv('speedcat.csv')
#---------------------------Data normalization---------------------------------------
scaler1 = MinMaxScaler(feature_range=(0, 1))
spe = scaler1.fit_transform(spe)
spe = np.reshape(spe,(1008,1,1,1,7))



# load the new file
target = pd.read_csv('targetctrg2021.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
print(target.head())
print(target.shape)
# save
#spe.to_csv('speedcat.csv')
#---------------------------Data normalization---------------------------------------
scaler8 = MinMaxScaler(feature_range=(0, 1))
target=scaler8.fit_transform(target)
target=np.reshape(target,(1008,1))



#--------------Create the input data set------------------------------------------------------------
train_spe= spe


                 
test_spe= target


#the dataset was divided into two parts: the training dataset and the testing dataset
train_size = int(len(train_spe) * 0.80)
X1=train_spe[0:train_size,:]                 


Y1=test_spe[0:train_size,:]                


y1=Y1


X1_test=train_spe[train_size:,:]                 




Y1_test=test_spe[train_size:,:]                 


y1_test=Y1_test


look_back=1
flters_no=10
#------------learn spatio-temporal feature from the speed data-----------------------------------------
spe_input = Input(shape=(look_back,1,1,7))

spe_input1 = BatchNormalization()(spe_input)
layer4 = ConvLSTM2D(filters=flters_no, kernel_size=(3, 3),padding='same',kernel_regularizer = regularizers.l2(0.01), bias_regularizer = regularizers.l2(0.01),data_format='channels_last', return_sequences=False)(spe_input1 )

flat1 = Flatten()(layer4)

              

               
#------------Combining the spatio-temporal information using a fusion layer----------------------------------
#merged_output = keras.layers.concatenate([layer2, layer4, layer6, layer8, layer10, layer12, layer14])
merged_output = flat1
#out = keras.layers.Dense(128)(merged_output)
out = keras.layers.Dense(1)(merged_output)
model = Model(inputs=spe_input, outputs=out)
model.compile(loss='mean_squared_error', optimizer='Adamax')
start = time.time()
#-----------------------Record training history---------------------------------------------------------------
train_history = model.fit(X1, y1, epochs=150, batch_size=32, verbose=1,validation_data=(X1_test, y1_test))
print(X1.shape)
print(y1.shape)
print(X1)
print(y1)

loss = train_history.history['loss']
val_loss=train_history.history['val_loss']
end = time.time()
print (end-start)
plt.plot(train_history.history['loss'], label='train')
plt.plot(train_history.history['val_loss'], label='test')
plt.legend()
plt.xlabel("Number of epochs")
plt.ylabel("Loss")
plt.show()
#--------------------------------Make prediction----------------------------------------------------------------
y1_pre = model.predict(X1_test)

print(y1_test.shape)
print(y1_pre.shape)

y1_test1 = scaler8.inverse_transform(y1_test)
y1_pre1 = scaler8.inverse_transform(y1_pre)
y1_test2=np.reshape(y1_test1,(1,202))
y1_pre2=np.reshape(y1_pre1,(1,202))



MSE=mean_squared_error(y1_pre1,y1_test1)
MAE=mean_absolute_error(y1_pre1,y1_test1)
# save the prediction values and the real values
np.savetxt( 'test1.txt',y1_test1)
# save the prediction values and the real values
np.savetxt( 'pre1.txt',y1_pre1 )
#--------------------------------Calculate evaluation index-----------------------------------------------------
mape= np.mean((abs(y1_test1- y1_pre1)) /y1_test1)
rmse=(y1_test1- y1_pre1)*(y1_test1- y1_pre1)
rm=np.sum(rmse)
RMSE=math.sqrt(rm/(rmse.size))
ape2=(abs(y1_test1- y1_pre1)) /y1_test1
ape22=ape2*ape2
summape2=np.sum(ape2)
summape22=np.sum(ape22)
len2=ape2.size
vape=math.sqrt((len2*summape22-summape2*summape2)/(len2*(len2-1)))
ec=(math.sqrt((np.sum((y1_test1- y1_pre1)**2))/len(y1_test1)))/(math.sqrt((np.sum(y1_test1**2))/len(y1_test1))+math.sqrt((np.sum(y1_pre1**2))/len(y1_test1)))
tic = (math.sqrt( (np.sum((y1_test1- y1_pre1)**2)) / len(y1_test1) )) / (math.sqrt((np.sum((y1_test1)**2)) / len(y1_test1) ) + math.sqrt((np.sum((y1_test1)**2)) / len(y1_test1)))
cc = np.corrcoef(y1_test2, y1_pre2)
print('MAE:', MAE)
print('MSE:', MSE)
#print('RMSE:', RMSE)
print('MAPE' , mape)
#print('EC' , ec)
#print('TIC' , tic)
print('cc' , cc)
print('Train Score: %.4f VAPE' % (vape))
