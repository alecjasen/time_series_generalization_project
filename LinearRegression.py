import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import sklearn.metrics as mt

from pandas_datareader import data as pdr
import yfinance as yf

##10 and 20 and 40 days Lookback
##RMSQError
##Average 0,5,10,20
##Remember 5 days is a week
##Regress over Next Period or 20 days
#Predicting 20 days by rolling window

def down_sample(data, period):
    if per==0:
        return data
    else:
        return data.rolling(period).mean()[period-1:-1].to_numpy()

def Test_RMSE(model,act_y,initial_test):
    pred_y = np.zeros(act_y.shape)
    test = initial_test
    for i in range(pred_y.size):
        pred_y[i] = model.predict(test)
        test[0,:] = np.append(test[0,1:],pred_y[i])
    return mt.mean_squared_error(act_y,pred_y,squared=False),pred_y

def Test_sMAPE(act_y,pred_y):
    Num = abs(act_y-pred_y)
    Den = (abs(act_y)+abs(pred_y))*.5
    Loss = np.mean(Num/Den)*100

    return Loss

yf.pdr_override() # <== that's all it takes :-)
# download dataframe
Data = pdr.get_data_yahoo("^GSPC", start="2000-01-01", end="2021-01-01")['Adj Close']

# Data = pd.read_csv("SPData")["Adj Close"]
EN_model_dict = {}
LR_model_dict = {}
l_1ratio = [.01,.05,.1,.25, .5, .7, .9, .95 ,99, 1]
Test_Error = np.empty((2,4,3))
Train_Error = np.empty((2,4,3))
Test_Error.fill(np.nan)
Train_Error.fill(np.nan)
Per_Test_Error = np.empty((2,4,3))
Per_Train_Error = np.empty((2,4,3))
Per_Test_Error.fill(np.nan)
Per_Train_Error.fill(np.nan)
m=-1
for per in [0,5,10,20]:
    n=-1
    m+=1
    data = down_sample(Data,per)
    for lkbk in [10,20,40]:
        n+=1
        string = str(per)+',' +str(lkbk)
        test_size = 20
        if per>=lkbk:
            continue
        elif per != 0:
            lkbk //= per
            test_size//=per

        ##Prepare Data
        X_train = np.zeros((data.size-lkbk-test_size,lkbk))
        y_train = np.zeros((data.size-lkbk-test_size,))
        init_X_test = np.zeros((1, lkbk))
        y_test = np.zeros((test_size,))
        for i in range(data.size-lkbk-test_size):
            X_train[i,:] = data[i:i+lkbk]
            y_train[i] = data[i+lkbk]

        init_X_test[0,:] = data[-1 - lkbk - test_size +1:-1-test_size+1]
        y_test[:] = data[-1 - test_size +1:]
        print(X_train.shape)
        print(y_test.shape)
        ## Do Regression
        LR_model_dict[string]= lm.LinearRegression()
        EN_model_dict[string] = lm.ElasticNetCV(l1_ratio=l_1ratio,tol=1e-6,max_iter=10000,cv=10)
        LR_model_dict[string].fit(X_train,y_train)
        EN_model_dict[string].fit(X_train,y_train)

        Train_Error[0,m,n] = mt.mean_squared_error(y_train,LR_model_dict[string].predict(X_train),squared=False)
        Train_Error[1,m,n] = mt.mean_squared_error(y_train,EN_model_dict[string].predict(X_train),squared=False)
        Test_Error[0,m,n],pred_yLR=Test_RMSE(LR_model_dict[string],y_test,init_X_test)
        Test_Error[1,m,n],pred_yEN = Test_RMSE(EN_model_dict[string], y_test, init_X_test)
        Per_Train_Error[0, m, n] = Test_sMAPE(y_train, LR_model_dict[string].predict(X_train))
        Per_Train_Error[1, m, n] = Test_sMAPE(y_train, EN_model_dict[string].predict(X_train))
        Per_Test_Error[0, m, n] = Test_sMAPE(y_test, pred_yLR)
        Per_Test_Error[1, m, n] = Test_sMAPE(y_test, pred_yEN)


