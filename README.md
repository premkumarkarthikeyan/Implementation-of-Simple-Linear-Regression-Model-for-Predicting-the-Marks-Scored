# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Libraries: Import essential libraries for data manipulation, numerical operations, plotting, and regression analysis.
2. Load and Explore Data: Load a CSV dataset using pandas, then display initial and final rows to quickly explore the data's structure.
3.Prepare and Split Data: Divide the data into predictors (x) and target (y). Use train_test_split to create training and testing subsets for model building and evaluation.
4. Train Linear Regression Model: Initialize and train a Linear Regression model using the training data.
5. Visualize and Evaluate: Create scatter plots to visualize data and regression lines for training and testing. Calculate Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) to quantify model performance.


## Program:

Program to implement the simple linear regression model for predicting the marks scored.
#### Developed by: SAKTHISWAR S
#### RegisterNumber: 212222230127

```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df = pd.read_csv('/content/student_scores.csv')
df.head()

df.tail()

X=df.iloc[:,:-1].values
X

y=df.iloc[:,1].values
y

X=df.iloc[:,:-1].values
Y=df.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

Y_pred
Y_test

plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color="green")
plt.plot(X_test,regressor.predict(X_test),color="black")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)

```

## Output:
### Head:
![image](1.png)
### Tail:
![image](2.png)
### Array value of X:
![image](3.png)
### Array value of Y:
![image](4.png)
### Values of Y prediction:
![image](5.png) 
### Array values of Y test:
![image](6.png)
### Training Set Graph:
![image](7.png)
### Test Set Graph:
![image](8.png)
### Values of MSE, MAE and RMSE:

![image](9.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
