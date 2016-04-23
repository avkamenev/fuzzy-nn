import pandas as pd
from dateutil.relativedelta import relativedelta

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
CREATE_TEST_SPLIT = False


if CREATE_TEST_SPLIT:
    data = pd.read_csv('data/data_3k.csv', delimiter=',', index_col = False)
    data = data.drop(np.where(data['y']==0)[0])
    indices = range(len(data))
    np.random.shuffle(indices)
    sep_index = int(0.8*len(data))
    training, test = indices[:sep_index], indices[sep_index:]
    train_data = data.iloc[training]
    test_data = data.iloc[test]
    train_data.to_csv('data/train.csv', index=False)
    test_data.to_csv('data/test.csv', index=False)


#times = [row.split()[1].split(':') for row in data.iloc[:,0]]
#times = [int(row[0])*60 + int(row[1]) for row in times]
#data.iloc[:,0] = times

train_data = pd.read_csv('data/train.csv', delimiter=',', index_col = False)
x = train_data.iloc[:,:4].values
test_data = pd.read_csv('data/test.csv', delimiter=',', index_col = False)
x_test = test_data.iloc[:,:4].values

min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(np.row_stack((x,x_test)))

x = min_max_scaler.transform(x)
x_test = min_max_scaler.transform(x_test)

y = train_data.iloc[:,4].values
y_test = test_data.iloc[:,4].values


rfc = RandomForestRegressor(n_estimators=500)
rfc = rfc.fit(x, y)
#round(sum(abs(rfc.predict(x) - y)/abs(y))/len(y),4)
#round(sum(abs(rfc.predict(x_test) - y_test)/abs(y_test))/len(y_test),4)
print np.sqrt(np.mean((rfc.predict(x) - y)**2))
print np.sqrt(np.mean((rfc.predict(x_test) - y_test)**2))





home_depot = pd.read_csv('/home/andrey/Kaggle/home-depot/dataset/all_good_features/all_good_features_train.csv')

indices = range(len(home_depot))
np.random.shuffle(indices)
#home_depot = home_depot.iloc[indices[:50000]]

home_depot.iloc[0]
home_depot.shape

x = home_depot[['words_in_title','words_in_descr','number_in_query','query_len','title_len','descr_len','ratio_title','ratio_descr', 'sim_with_title_w2v', 'sim_with_descr_w2v', 'sim_with_title_w2v_title_descr', 'sim_with_descr_w2v_title_descr']].values
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(np.row_stack(x))
y = home_depot['relevance'].values



#-----------------------------------------

x = train_data[['words_in_title','words_in_descr','number_in_query','query_len','title_len','descr_len','ratio_title','ratio_descr', 'sim_with_title_w2v', 'sim_with_descr_w2v', 'sim_with_title_w2v_title_descr', 'sim_with_descr_w2v_title_descr']].values
x_test = test_data[['words_in_title','words_in_descr','number_in_query','query_len','title_len','descr_len','ratio_title','ratio_descr', 'sim_with_title_w2v', 'sim_with_descr_w2v', 'sim_with_title_w2v_title_descr', 'sim_with_descr_w2v_title_descr']].values

min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(np.row_stack((x,x_test)))

x = min_max_scaler.transform(x)
x_test = min_max_scaler.transform(x_test)

y = train_data['relevance'].values
y_test = test_data['relevance'].values


#-----------------------------------------------------
# Balanced sample
#-----------------------------------------------------
np.random.seed(1234)
y = home_depot['relevance'].values
values = [1, 1.33, 1.67, 2, 2.33, 2.67, 3]
part_size = int(len(y)*np.mean(y==1))
train_part = int(part_size*0.8)
valid_part = part_size - train_part
train_indices = np.zeros(train_part*len(values))
valid_indices = np.zeros(valid_part*len(values))

for n in range(len(values)):
    indices = list(np.where(y==values[n]))[0]
    np.random.shuffle(indices)
    train_indices[(n*train_part):((n+1)*train_part)] = indices[:train_part]
    valid_indices[(n*valid_part):((n+1)*valid_part)] = indices[train_part:(train_part+valid_part)]
train_indices = train_indices.astype(int)
valid_indices = valid_indices.astype(int)


print np.mean(y[valid_indices]==1)*100
print np.mean(y[valid_indices]==1.33)*100
print np.mean(y[valid_indices]==1.67)*100
print np.mean(y[valid_indices]==2)*100
print np.mean(y[valid_indices]==2.33)*100
print np.mean(y[valid_indices]==2.67)*100
print np.mean(y[valid_indices]==3)*100



x = home_depot.iloc[train_indices].drop('relevance', axis=1).values
y = home_depot['relevance'].values[train_indices]
rfc = RandomForestRegressor(n_estimators=1000)
rfc = rfc.fit(x, y)
#round(sum(abs(rfc.predict(x) - y)/abs(y))/len(y),4)
#round(sum(abs(rfc.predict(x_test) - y_test)/abs(y_test))/len(y_test),4)
print np.sqrt(np.mean((rfc.predict(x) - y)**2))

x_valid = home_depot.iloc[valid_indices].drop('relevance', axis=1).values
y_valid = home_depot['relevance'].values[valid_indices]
print np.sqrt(np.mean((rfc.predict(x_valid) - y_valid)**2))

