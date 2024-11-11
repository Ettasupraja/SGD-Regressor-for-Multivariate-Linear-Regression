# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Start
2.Data Preparation
3.Hypothesis Definition
4.Cost Function
5.Parameter Update Rule
6.Iterative Training
7.Model Evaluation
8.End 
```
## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: ETTA SUPRAJA
RegisterNumber:  212223220022
*/
```
```
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
```

![image](https://github.com/user-attachments/assets/dbe8b2e4-c28a-4884-b8a2-7f4c76de4e61)


```
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
print(df.head())
```

![image](https://github.com/user-attachments/assets/587eeb60-9285-4923-829a-b62d971c8bd1)

```
df.info()
```

![image](https://github.com/user-attachments/assets/20964e82-1a7e-4ff8-9d6a-87c137d8326c)

```
X=df.drop(columns=['AveOccup','target'])
X.info()
```

![image](https://github.com/user-attachments/assets/aa82595b-63fa-41b1-bff5-b53ee393ebed)

```
Y=df[['AveOccup','target']]
Y.info()
```
![image](https://github.com/user-attachments/assets/d020a24b-fb66-4d14-a905-5254be2d3357)

```
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
x.head()
```
![image](https://github.com/user-attachments/assets/fa10a329-ca9b-4749-aa1a-ffff96e8a54c)

```
scaler_x=StandardScaler()
scaler_y=StandardScaler()
x_train=scaler_x.fit_transform(x_train)
x_test=scaler_x.transform(x_test)
y_train=scaler_y.fit_transform(y_train)
y_test=scaler_y.transform(y_test)
print(x_train)
```

![image](https://github.com/user-attachments/assets/cceee29a-5dad-4139-8697-2b68bcbf8163)

```
sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(x_train,y_train)
```

![image](https://github.com/user-attachments/assets/b1f8f9f4-b2e1-42fe-ac5e-111a6b8a47db)

```
y_pred=multi_output_sgd.predict(x_test)
y_pred=scaler_y.inverse_transform(y_pred)
y_test=scaler_y.inverse_transform(y_test)
mse=mean_squared_error(y_test,y_pred)
print("Mean Squared Error:",mse)
```

![image](https://github.com/user-attachments/assets/75118b0e-96e1-4815-8603-94973db0a2b8)

```
print("\nPredictions:\n", y_pred[:5])
```

![image](https://github.com/user-attachments/assets/4ccc1381-ebdd-45a7-b6cd-20e37b2dd35e)



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
