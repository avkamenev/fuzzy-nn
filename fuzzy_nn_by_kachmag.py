from numpy import linalg as LA
from itertools import product
import scipy


mfs = [[0,0.05], [0.1,0.05], [0.2,0.05], [0.3,0.05], [0.4,0.05], [0.5,0.05], [0.6,0.05], [0.7,0.05], [0.8,0.05], [0.9,0.05], [1,0.05]]

fRules = list(product(range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs))))

w_values = np.zeros((len(x), len(fRules)))
for n in range(len(fRules)):
    rule = fRules[n]
    w_value = np.zeros((len(x), len(rule)))
    for i in range(len(rule)):
        w_value[:,i] = scipy.stats.norm(mfs[rule[i]][0], mfs[rule[i]][1]).pdf(x[:,i])
    #w_value = np.sum(scipy.stats.norm(mfs[rule[i]][0], mfs[rule[i]][1]).pdf(x), axis=1)
    w_values[:,n] = np.max(w_value, axis=1)

beta = w_values/np.array([np.sum(w_values, axis=1)]).T
x_with_one = np.column_stack((np.ones(len(x)).T, x))

c = np.random.rand((x.shape[1]+1)*len(fRules),1)-0.5
#c = np.random.rand(number_of_rules, x.shape[1]+1)
#c = np.zeros(((x.shape[1]+1)*len(fRules),1))

x_model = np.zeros((len(x), (x.shape[1]+1)*len(fRules)))
for i in range(len(x)):
    x_model[i,:] = np.reshape(np.array([x_with_one[i,:]]).T.dot(np.array([beta[i,:]])), ( (x.shape[1]+1)*len(fRules)))

old_error=1
for loop_numbers in range(300):
    if loop_numbers % 10 == 0:
        print str(loop_numbers) + ':'
        y_model = np.reshape(x_model.dot(c), y.shape)
        error = round(np.sum(abs(y_model - y)/abs(y))/len(y),5)
        print str(error) + ' with improvement: ' + str(old_error-error)
        old_error = error
    for t in range(len(x)):
        #y_from_rule = x_with_one[t,:].dot(c.T)
        #y_model = np.sum(beta[t,:] * y_from_rule)

        #x_hat = np.reshape(np.array([x_with_one[t,:]]).T.dot(np.array([beta[t,:]])), ((x.shape[1]+1)*len(fRules),1))
        y_model = x_model[t,:].dot(c)
        c = c + 0.1*( (y[t] - y_model) / LA.norm(x_model[t,:])**2 ) * np.reshape(x_model[t,:], c.shape)




########
# TEST #
########
w_values = np.zeros((len(x_test), len(fRules)))
for n in range(len(fRules)):
    rule = fRules[n]
    w_value = np.zeros((len(x_test), len(rule)))
    for i in range(len(rule)):
        w_value[:,i] = scipy.stats.norm(mfs[rule[i]][0], mfs[rule[i]][1]).pdf(x_test[:,i])
    w_values[:,n] = np.max(w_value, axis=1)

beta_test = w_values/np.array([np.sum(w_values, axis=1)]).T
x_test_with_one = np.column_stack((np.ones(len(x_test)).T, x_test))
x_test_model = np.zeros((len(x_test), (x_test.shape[1]+1)*len(fRules)))
for i in range(len(x_test)):
    x_test_model[i,:] = np.reshape(np.array([x_test_with_one[i,:]]).T.dot(np.array([beta_test[i,:]])), ( (x_test.shape[1]+1)*len(fRules)))

y_test_model = np.reshape(x_test_model.dot(c), y_test.shape)
print round(np.sum(abs(y_test_model - y_test)/abs(y_test))/len(y_test),5)



