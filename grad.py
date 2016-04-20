from itertools import product
import scipy


number_of_separations = 3
step_for_mf = 1./(number_of_separations-1)
#mfs = [[0,0.05], [0.1,0.05], [0.2,0.05], [0.3,0.05], [0.4,0.05], [0.5,0.05], [0.6,0.05], [0.7,0.05], [0.8,0.05], [0.9,0.05], [1,0.05]]
#mfs = [[0,0.05], [0.2,0.05], [0.4,0.05], [0.6,0.05], [0.8,0.05], [1,0.05]]
mfs = [[0,0.05], [step_for_mf,0.05], [1,0.05]]

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


x_model = np.zeros((len(x), (x.shape[1]+1)*len(fRules)))
for i in range(len(x)):
    x_model[i,:] = np.reshape(np.array([x_with_one[i,:]]).T.dot(np.array([beta[i,:]])), ( (x.shape[1]+1)*len(fRules)))

c = np.random.rand((x.shape[1]+1)*len(fRules),1)-0.5

alfa = 0.1

#grad = - 1/len(x) * np.array([np.sum(x_model/(np.array([y]).T), axis=0)]).T
while True:
    y_model = np.reshape(x_model.dot(c), y.shape)
    grad = 1./len(x) * np.array([np.sum(np.array([y_model-y]).T * x_model, axis=0)]).T
    c = c - alfa * grad

    y_model = np.reshape(x_model.dot(c), y.shape)
    error = np.sqrt(np.mean((y_model - y)**2))
    print str(round(error,5))



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

print np.sqrt(np.mean((y_test_model - y_test)**2))


