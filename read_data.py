import pandas as pd
from dateutil.relativedelta import relativedelta
import timeit
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





#------------------------------
# scaling
#------------------------------

min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(np.row_stack((x,x_test)))
x = min_max_scaler.transform(x)
x_test = min_max_scaler.transform(x_test)


#------------------------------
# RF
#------------------------------
rfc = RandomForestRegressor(n_estimators=100)
start = timeit.default_timer()
rfc = rfc.fit(x, y)
stop = timeit.default_timer()
print (stop - start)/60.
#round(sum(abs(rfc.predict(x) - y)/abs(y))/len(y),4)
#round(sum(abs(rfc.predict(x_test) - y_test)/abs(y_test))/len(y_test),4)
print np.sqrt(np.mean((rfc.predict(x) - y)**2))
print np.sqrt(np.mean((rfc.predict(x_test) - y_test)**2))

print np.sqrt(np.mean((rfc.predict(x) - y)/y))
print np.sqrt(np.mean(abs((rfc.predict(x_test) - y_test)/y_test)))


#------------------------------
#????
#------------------------------
#x = home_depot.iloc[train_indices].drop('relevance', axis=1).values
x = home_depot[['words_in_title','words_in_descr','number_in_query','query_len','title_len','descr_len','ratio_title','ratio_descr', 'sim_with_title_w2v', 'sim_with_descr_w2v', 'sim_with_title_w2v_title_descr', 'sim_with_descr_w2v_title_descr']].iloc[train_indices].values

y = home_depot['relevance'].values[train_indices]
rfc = RandomForestRegressor(n_estimators=500)
rfc = rfc.fit(x, y)
#round(sum(abs(rfc.predict(x) - y)/abs(y))/len(y),4)
#round(sum(abs(rfc.predict(x_test) - y_test)/abs(y_test))/len(y_test),4)
print np.sqrt(np.mean((rfc.predict(x) - y)**2))

#x_valid = home_depot.iloc[valid_indices].drop('relevance', axis=1).values
# x_valid = home_depot[['words_in_descr','number_in_query','query_len','title_len','descr_len','ratio_title','ratio_descr', 'sim_with_title_w2v', 'sim_with_descr_w2v', 'sim_with_title_w2v_title_descr', 'sim_with_descr_w2v_title_descr']].iloc[valid_indices].values
# y_valid = home_depot['relevance'].values[valid_indices]

#val_data = home_depot.drop(list(train_indices))
val_data = home_depot[['words_in_title','words_in_descr','number_in_query','query_len','title_len','descr_len','ratio_title','ratio_descr', 'sim_with_title_w2v', 'sim_with_descr_w2v', 'sim_with_title_w2v_title_descr', 'sim_with_descr_w2v_title_descr', 'relevance']].drop(list(train_indices))

x_valid = val_data.drop('relevance', axis=1).values
y_valid = val_data['relevance'].values

print np.sqrt(np.mean((rfc.predict(x_valid) - y_valid)**2))


min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(np.row_stack((x,x_valid)))

x = min_max_scaler.transform(x)
x_test = min_max_scaler.transform(x_valid)
y_test = y_valid

