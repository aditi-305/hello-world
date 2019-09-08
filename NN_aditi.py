import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation,Layer,Lambda
from keras.utils import np_utils

from sklearn.model_selection import train_test_split

concrete = pd.read_csv("C:\\Users\\Aditi\\Desktop\\data_science _data _set\\assignments\\Neural networks\\concrete.csv")


def prep_model(hidden_dim):
    model = Sequential()
    for i in range(1,len(hidden_dim)-1):
        if (i==1):
            model.add(Dense(hidden_dim[i],input_dim=hidden_dim[0],kernel_initializer="normal",activation="relu"))
        else:
            model.add(Dense(hidden_dim[i],activation="relu"))
    model.add(Dense(hidden_dim[-1]))
    model.compile(loss="mean_squared_error",optimizer="adam",metrics = ["accuracy"])
    return (model)

column_names = list(concrete.columns)
predictors = column_names[0:8]
target = column_names[8]

first_model = prep_model([8,50,1])
first_model.fit(np.array(concrete[predictors]),np.array(concrete[target]),epochs=900)
pred_train = first_model.predict(np.array(concrete[predictors]))
pred_train = pd.Series([i[0] for i in pred_train])
rmse_value = np.sqrt(np.mean((pred_train-Concrete[target])**2))
import matplotlib.pyplot as plt
plt.plot(pred_train,Concrete[target],"bo")
np.corrcoef(pred_train,Concrete[target]) # we got high correlation 
