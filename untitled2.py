import pandas as pd
import numpy as np

df = pd.read_csv("/w1.student_scores.csv")

df.head()

from matplotlib import pyplot as plt
plt.figure(figsize=(6,4))
plt.plot(df['Hours'], df['Scores'],'ro')
plt.title('Students scores vs hours')
plt.xlabel('Hours')
plt.ylabel('Scores')

plt.show()

xmean= np.mean(df['Hours'])
ymean= np.mean(df['Scores'])

df['xycov'] = (df['Hours'] - xmean) * (df['Scores'] - ymean)
df['xvar'] = (df['Hours'] - xmean) **2

slope = df ['xycov'] . sum() /df['xvar'] .sum()
intercept = ymean - (slope * xmean)
print (f'slope ={slope}')
print (f'intercept ={intercept}')

"""Making Prediction"""

intercept + slope*2

Scores_predicted = intercept + slope* df['Hours']

from matplotlib import pyplot as plt
plt.figure(figsize=(6,4))
plt.plot(df['Hours'],Scores_predicted )
plt.plot(df['Hours'], df['Scores'],'ro')
plt.title('Actual vs Predicted')
plt.xlabel('Hours')
plt.ylabel('Scores')

plt.show()

df = pd.read_csv("/content/sample_data/w1_petrol_consumption-220925-152435.csv")

df.head(5)

df.shape

df.describe().round(2).T

from matplotlib import pyplot as plt
plt.figure(figsize=(6,4))
plt.plot(df['Petrol_tax'], df['Petrol_Consumption'],'ro')
plt.title('Petrol_tax vs Petrol_Consumption')
plt.xlabel('Petrol_tax')
plt.ylabel('Petrol_consumption')

plt.show()

from matplotlib import pyplot as plt
plt.figure(figsize=(6,4))
plt.plot(df['Average_income'], df['Petrol_Consumption'],'ro')
plt.title('Average_income vs Petrol_Consumption')
plt.xlabel('Average_income')
plt.ylabel('Petrol_consumption')

plt.show()

y = df ['Petrol_Consumption']
x = df [['Average_income' , 'Paved_Highways' , 'Population_Driver_licence(%)' , 'Petrol_tax'] ]

from sklearn.model_selection import train_test_split

x_train ,  x_test, y_train , y_test = train_test_split (x,y,test_size=0.2 , random_state=42)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression ()
regressor.fit (x_train, y_train)

regressor.intercept_
regressor.coef_

y_pred = regressor.predict(x_test)

results = pd.DataFrame ({'Actual': y_test, 'Predicted': y_pred})
print(results)

from sklearn.metrics import mean_absolute_error, mean_squared_error
MAE = mean_absolute_error (y_test, y_pred)
MSE = mean_squared_error (y_test, y_pred)
RMSE = np.sqrt (MSE)

print(f'Mean Absolute Error : {MAE: .2f}')
print(f'Mean Squared Error : {MSE: .2f}')
print(f'Root Mean Squared Error : {RMSE: .2f}')

regressor.score (x_test,y_test)

regressor.score (x_train,y_train)
