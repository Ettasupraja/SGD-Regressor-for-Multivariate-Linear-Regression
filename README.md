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

![image](https://github.com/user-attachments/assets/f1070a17-9af8-4790-a309-7a7d3e8c58a2)


df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
print(df.head())

![image](https://github.com/user-attachments/assets/7001e5ef-4a4f-48bc-918e-3aeb52d26ed5)


df.info()

![image](https://github.com/user-attachments/assets/e5abc107-91d8-4d29-b869-dbed3aac1502)


X=df.drop(columns=['AveOccup','target'])
X.info()

![image](https://github.com/user-attachments/assets/509f40e0-680c-49bb-a0de-35b0a9031502)


Y=df[['AveOccup','target']]
Y.info()

![image](https://github.com/user-attachments/assets/d23f0375-a13e-4eb8-abe8-af438ab1d5fb)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
x.head()

![image](https://github.com/user-attachments/assets/ad83fda3-71cd-4ec6-992a-5c429a1fe209)

scaler_x=StandardScaler()
scaler_y=StandardScaler()
x_train=scaler_x.fit_transform(x_train)
x_test=scaler_x.transform(x_test)
y_train=scaler_y.fit_transform(y_train)
y_test=scaler_y.transform(y_test)
print(x_train)

![image](https://github.com/user-attachments/assets/08eeaa78-14d2-4870-9491-03c8b819428d)

sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(x_train,y_train)

![image](https://github.com/user-attachments/assets/6e94baee-1983-4ab9-a699-379cdfd930b8)

y_pred=multi_output_sgd.predict(x_test)
y_pred=scaler_y.inverse_transform(y_pred)
y_test=scaler_y.inverse_transform(y_test)
mse=mean_squared_error(y_test,y_pred)
print("Mean Squared Error:",mse)

![image](https://github.com/user-attachments/assets/2c068bb0-5076-4f08-9c2e-d586b1312254)

print("\nPredictions:\n", y_pred[:5])

![Uploading image.png…]()

```

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
