import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, LSTM, Conv1D,Dropout,Bidirectional
from timeit import default_timer as timer
from sklearn.metrics import median_absolute_error
from keras.layers import *
from keras.models import *
from sklearn.model_selection import train_test_split
from keras import initializers
import keras.backend as K
import tensorflow as tf
import os
from sklearn.metrics import r2_score, mean_squared_error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
path = fr"F:\datasetpath.csv"
createVar = locals()
def datapro(df):
    df['RT_log'] = np.log10(df.RT.values)
    for i in df.keys():
        df[i + "grad"] = np.gradient(df[i].values)
    df['grad']=np.gradient(df['DEN'].values)
    return df

def create_dataset(dataset, back=1, pad=True):
    if pad:
        dataset_pad = np.pad(dataset, ((back, back), (0, 0)), mode='edge')
    else:
        dataset_pad = dataset
    dataX = []
    for i in range(len(dataset_pad) - 2 * back):
        a = dataset_pad[i:(i + 2 * back + 1), :]
        dataX.append(a)
    return np.array(dataX)

def attention(inputs):
    input_dim = int(inputs.shape[2])
    aT = Permute((2, 1))(inputs)
    aT = Dense(TIME_STEPS, activation='softmax')(aT)
    aT = Lambda(lambda x: K.mean(x, axis=1))(aT)
    aT = RepeatVector(input_dim)(aT)
    aT = Permute((2,1),name='attention_vec1')(aT)
    aF = Dense(input_dim, activation='softmax',name='attention_vec2')(inputs)
    aF = Lambda(lambda x: K.mean(x, axis=1))(aF)
    aF = RepeatVector(TIME_STEPS)(aF)
    a_probs = Multiply()([aT,aF])
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

def load(): #data loading
    dfs = []
    for i in range(1, 13):
        dfs.append(pd.read_csv(path))
        dfs[i - 1].columns = ['DEPTH','DEN', 'DTC','DTS' ,'GR', 'RT','zone']

    for i in range(0, 12):
        createVar['Well' + str(i + 1)] = pd.DataFrame(dfs[i])
        createVar['Well' + str(i + 1)] = createVar['Well' + str(i + 1)].replace(['-99999', -99999], np.nan)
        createVar['Well' + str(i + 1)][createVar['Well' + str(i + 1)] < 0] = np.nan
        createVar['Well' + str(i + 1)].dropna(inplace=True)
        createVar['Well' + str(i + 1)].reset_index(drop=True, inplace=True)

    df1 = Well1.copy()
    df2 = Well2.copy()
    df3 = Well3.copy()
    df4 = Well4.copy()

    data_1 = pd.concat([df1[df1['zone']== 1],df3[df3['zone']== 1],df2[df2['zone']== 1]])
    data_2 = pd.concat([df1[df1['zone']== 2],df3[df3['zone']== 2],df2[df2['zone']== 2]])
    data_3 = pd.concat([df1[df1['zone']== 3],df3[df3['zone']== 3],df2[df2['zone']== 3]])

    MAX = pd.Series({'DEN': , 'DTC': , 'GR': , 'RT_log':}) //Enter the maximum and minimum values
    MIN = pd.Series({'DEN': , 'DTC': , 'GR': , 'RT_log':})

    Data_1= datapro(data_1)
    Data_2= datapro(data_2)
    Data_3= datapro(data_3)

    MData_1 = (Data_1[['DEN','DTC','GR','RT_log']] - MIN)/(MAX-MIN)
    MData_2 = (Data_2[['DEN','DTC','GR','RT_log']] - MIN)/(MAX-MIN)+1
    MData_3 = (Data_3[['DEN','DTC','GR','RT_log']] - MIN)/(MAX-MIN)+2

    TrainSet = pd.concat([MData_1,MData_2,MData_3])
    Y = pd.concat([Data_1,Data_2,Data_3])
    X_TrainSet = TrainSet[['DEN','DTC','GR','RT_log']]
    Y_TrainSet = Y[['DTS']]

    x_train, x_test, y_train, y_test = train_test_split(X_TrainSet, Y_TrainSet, test_size=0.3, random_state=100)  
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    x_train = create_dataset(x_train,2)
    x_test = create_dataset(x_test,2)
    TestWell=datapro(df4)

    return x_train, y_train, x_test, y_test,TestWell, MAX, MIN


x_train, y_train, x_test, y_test,TestWell,MAX,MIN= load()

output_dim = 1  
batch_size =32
epochs = 600 
filiter = 128

TIME_STEPS = 5
INPUT_DIM = 4  


kernel_initializer = initializers.glorot_uniform(seed=0)
recurrent_initializer = initializers.Orthogonal(gain=1.0, seed=0)
inputs = Input(shape=(TIME_STEPS, INPUT_DIM))
x = Conv1D(filters = filiter, kernel_size = 5, activation = 'relu',kernel_initializer="uniform")(inputs)
x = MaxPooling1D(pool_size = 1)(x)
x = Flatten()(x)
attention = attention(inputs)
lstm_out = LSTM(128,activation='tanh',dropout=0.3)(attention)
merge = Concatenate(axis=1)([x,lstm_out])
output = Dense(1,activation='linear',kernel_initializer="uniform",name='out')(merge)
model = Model(inputs=inputs, outputs=output)

model.compile(loss='mean_squared_error', optimizer='adam',metrics= ['mae'])
start = timer()
lr_reduce= tf.keras.callbacks.ReduceLROnPlateau('val_loss',patience=3,factor=0.5,min_lr=0.00001)
history = model.fit(x_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(x_test,y_test),callbacks=[lr_reduce])#
end = timer()
print("Traing time is:",end - start)

epochs=range(len(history.history['loss']))
plt.figure()
plt.plot(epochs,history.history['loss'],'b',label='Training loss')
plt.plot(epochs,history.history['val_loss'],'b--',label='Validation val_loss')
plt.title('Traing and Validation loss')
plt.legend()
plt.show()

outputloss = pd.DataFrame()
outputloss['loss'] = history.history['loss']
outputloss['val_loss'] = history.history['val_loss']
outputloss.to_csv('lossCNN.csv')

def wucha(y,predict):
    RMSE= np.sqrt(mean_squared_error(y, predict))
    MSE=mean_squared_error(y, predict)
    MAE = median_absolute_error(y, predict)
    r2_score_test=r2_score(y, predict)
    print(" MSE{:.4}, MAE{:.2},RMSE{:.4},R2:{}".format(MSE,MAE,RMSE,r2_score_test))


def result_plot(y_predict, y_real):
    plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
    plt.subplot(2, 1, 1)
    plt.plot(y_real)
    plt.plot(y_predict, '--')
    plt.legend(['True', 'Predicted'])
    plt.xlabel('Sample')
    plt.ylabel('DTS')
    plt.title('DTS Prediction Comparison')
    plt.show()

train = model.predict(x_train)
wucha(y_train,train)
result_plot(train, y_train)

p_train = model.predict(x_test)
wucha(y_test,p_train)
result_plot(p_train, y_test)


