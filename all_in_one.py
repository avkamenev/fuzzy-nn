import pandas as pd
from dateutil.relativedelta import relativedelta
import timeit
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing


train = pd.read_csv('~/home-depot/dataset/good_ft_2_train.csv')
test = pd.read_csv('~/home-depot/dataset/good_ft_2_test.csv')

print train.shape
print test.shape
print 'READED!!'

feature_names = ['sim_with_title_w2v_title_descr',
'id.1',
'sim_with_descr_w2v',
'sim_with_title_w2v',
'sim_with_descr_w2v_title_descr',
'descr_len',
'title_len',
'search_descr_tfidf_sum',
'search_title_tfidf_min',
'query_len',
'ratio_title']

x = train[feature_names[:]].values
x_test = test[feature_names[:]].values
y = train['relevance'].values
y_test = test['relevance'].values

min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(np.row_stack((x,x_test)))
x = min_max_scaler.transform(x)
x_test = min_max_scaler.transform(x_test)


rfc = RandomForestRegressor(n_estimators=500)
start = timeit.default_timer()
rfc = rfc.fit(x, y)
stop = timeit.default_timer()
print (stop - start)/60.
print np.sqrt(np.mean((rfc.predict(x) - y)**2))
print np.sqrt(np.mean((rfc.predict(x_test) - y_test)**2))



#KACHMAG
print '!!!!!!!!!!!!!!!!!!!!!!!!!!!1KACHMAG'


from numpy import linalg as LA
from itertools import product
import scipy

#mfs = [[0,0.05], [0.2,0.05], [0.4,0.05], [0.6,0.05], [0.8,0.05], [1,0.05]]
mfs = [[0,0.05], [0.5,0.05], [1,0.05]]
#mfs = [[0,0.05], [1,0.05]]
#fRules = list(product(range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs)),range(len(mfs)),range(len(mfs)),range(len(mfs)),range(len(mfs)), range(len(mfs))))
# fRules = list(product(range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs)),
#                       range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs)),
#                       range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs))))
fRules = list(product(range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs)),
                      range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs)),
                      range(len(mfs)), range(len(mfs)), range(len(mfs))))

np.random.seed(1234)
c = (np.random.rand((x.shape[1]+1)*len(fRules),1)-0.5) * 2
x_with_one = np.column_stack((np.ones(len(x)).T, x))
y_model = np.zeros(len(x))

x_mf = np.zeros((len(mfs),x.shape[0],x.shape[1]))
for i in range(len(mfs)):
    x_mf[i] = scipy.stats.norm(mfs[i][0], mfs[i][1]).pdf(x)

old_error=10
error=2
loop_numbers=0
max_loops = 5
errors = np.zeros(max_loops)
while ((old_error-error)>=0.00001) & (error>0.03) & (loop_numbers!=max_loops):
    print 'Loop: '+str(loop_numbers)
    start = timeit.default_timer()
    for t in range(len(x)):
        if t%1000==0:
            print t
        # w_values = np.zeros(len(fRules))
        # for n in range(len(fRules)):
        #     rule = fRules[n]
        #     w_value_for_rule = np.zeros(len(rule))
        #     for m in range(len(rule)):
        #         w_value_for_rule[m] = x_mf[rule[m],t,m]
        #     # w_value_for_rule = x_mf[np.array(rule)[range(x.shape[1])], t, range(x.shape[1])]
        #     w_values[n] = np.max(w_value_for_rule)

        w_values = x_mf[np.array(fRules)[:,range(x.shape[1])], t, range(x.shape[1])]
        w_values = np.max(w_values, axis=1)
        beta_t = w_values/np.sum(w_values)
        x_model_t = np.reshape(np.array([x_with_one[t,:]]).T.dot(np.array([beta_t])), ((x.shape[1]+1)*len(fRules)))
        y_model[t] = x_model_t.dot(c)
        c = c + 0.1*( (y[t] - y_model[t]) / LA.norm(x_model_t)**2 ) * np.reshape(x_model_t, c.shape)

    old_error = error
    error = np.sqrt(np.mean((y_model - y)**2))
    errors[loop_numbers] = error
    print 'ERROR: ' + str(error)
    stop = timeit.default_timer()
    print (stop - start)/60.
    loop_numbers += 1


print errors



#GRAD

print '!!!!!!!!!!!!!!!!!!GRAD'




old_error=10
error=2
loop_numbers=0
max_loops = 10
alfa = 100
errors = np.zeros(max_loops)
while ((old_error-error)>=0.00001) & (error>0.03) & (loop_numbers!=max_loops):
    print 'Loop: '+str(loop_numbers)
    start = timeit.default_timer()
    for t in range(len(x)):
        if t%1000==0:
            print t
        w_values = x_mf[np.array(fRules)[:,range(x.shape[1])], t, range(x.shape[1])]
        w_values = np.max(w_values, axis=1)
        beta_t = w_values/np.sum(w_values)
        x_model_t = np.reshape(np.array([x_with_one[t,:]]).T.dot(np.array([beta_t])), ((x.shape[1]+1)*len(fRules)))
        y_model[t] = x_model_t.dot(c)

        grad = 1./len(x) * np.array([(y_model[t]-y[t]) * x_model_t]).T
        c = c - alfa * grad

    old_error = error
    error = np.sqrt(np.mean((y_model - y)**2))
    errors[loop_numbers] = error
    print 'ERROR: ' + str(error)
    stop = timeit.default_timer()
    print (stop - start)/60.
    loop_numbers += 1


print errors


out = pd.DataFrame(c)
out.columns = ['params']
out.to_csv('params.csv', index=None)
