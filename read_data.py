import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing

data = pd.read_csv('train1.csv', delimiter=';')

#times = [row.split()[1].split(':') for row in data.iloc[:,0]]
#times = [int(row[0])*60 + int(row[1]) for row in times]
#data.iloc[:,0] = times

x = data.iloc[:,1:5].values
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)
y = data.iloc[:,5].values


rfc = RandomForestRegressor(n_estimators=100)
rfc = rfc.fit(x, y)
''+str(round(sum(abs(rfc.predict(x) - y)/abs(y))/len(y)*100,2))+'%'


test_data = pd.read_csv('test1.csv', delimiter=';')
x_test = test_data.iloc[:,1:5].values
y_test = test_data.iloc[:,5].values
''+str(round(sum(abs(rfc.predict(x_test) - y_test)/abs(y_test))/len(y_test)*100,2))+'%'







