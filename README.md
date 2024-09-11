# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start
2.Data Preparation
3.Hypothesis Definition
4.Cost Function
5.Parameter Update Rule
6.Iterative Training
7.Model Evaluation
8.End 

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: ETTA SUPRAJA
RegisterNumber:  212223220022
*/

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data=fetch_california_housing()
print(data)

![Screenshot 2024-09-11 135010](https://github.com/user-attachments/assets/3f1ac0d8-627b-49d2-bc95-dd597c6feade)

df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
print(df.head())

![Screenshot 2024-09-11 135218](https://github.com/user-attachments/assets/c7474b7f-9124-4fdc-b485-898851d9fd5c)

df.info()

![Screenshot 2024-09-11 205035](https://github.com/user-attachments/assets/e5683613-932d-463d-bee9-d7d81df29d36)


X=df.drop(columns=['AveOccup','target'])
X.info()

![Screenshot 2024-09-11 205140](https://github.com/user-attachments/assets/1f7bbfe6-86cb-4616-92d7-a1bceecc1b6c)


Y=df[['AveOccup','target']]
Y.info()

![Screenshot 2024-09-11 205219](https://github.com/user-attachments/assets/d7c4edcc-eb57-4bfb-82b3-866f742465e0)


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
x.head()

![Screenshot 2024-09-11 205305](https://github.com/user-attachments/assets/fd4a700d-e9d3-49b6-8a95-9f845d225dce)

scaler_x=StandardScaler()
scaler_y=StandardScaler()
x_train=scaler_x.fit_transform(x_train)
x_test=scaler_x.transform(x_test)
y_train=scaler_y.fit_transform(y_train)
y_test=scaler_y.transform(y_test)
print(x_train)

![Screenshot 2024-09-11 205345](https://github.com/user-attachments/assets/59ab31a2-ad69-45fc-ac60-9856f6a0015c)


sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(x_train,y_train)

![Screenshot 2024-09-11 205430](https://github.com/user-attachments/assets/85223de9-6e6e-431e-a1b8-651e827a97ec)


y_pred=multi_output_sgd.predict(x_test)
y_pred=scaler_y.inverse_transform(y_pred)
y_test=scaler_y.inverse_transform(y_test)
mse=mean_squared_error(y_test,y_pred)
print("Mean Squared Error:",mse)

![Screenshot 2024-09-11 205502](https://github.com/user-attachments/assets/81a7f3b0-5e20-4bb8-b410-420cf603f73d)

print("\nPredictions:\n", y_pred[:5])

![Screenshot 2024-09-11 205545](https://github.com/user-attachments/assets/93107a9f-43c0-4924-adff-c31a5dc6c122)


```

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
