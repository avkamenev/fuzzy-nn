from itertools import product
import scipy


mfs = [[0,0.05], [0.2,0.05], [0.4,0.05], [0.6,0.05], [0.8,0.05], [1,0.05]]

fRules = list(product(range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs))))

w_values = np.zeros((len(x), len(fRules)))
for n in range(len(fRules)):
    rule = fRules[n]
    w_value = np.zeros((len(x), len(rule)))
    for i in range(len(rule)):
        w_value[:,i] = scipy.stats.norm(mfs[rule[i]][0], mfs[rule[i]][1]).pdf(x[:,i])
    w_values[:,n] = np.max(w_value, axis=1)

beta = w_values/np.array([np.sum(w_values, axis=1)]).T
x_with_one = np.column_stack((np.ones(len(x)).T, x))


x_model = np.zeros((len(x), (x.shape[1]+1)*len(fRules)))
for i in range(len(x)):
    x_model[i,:] = np.reshape(np.array([x_with_one[i,:]]).T.dot(np.array([beta[i,:]])), ( (x.shape[1]+1)*len(fRules)))

np.random.seed(1234)
c = (np.random.rand((x.shape[1]+1)*len(fRules),1)-0.5) * 2

alfa = 30

y_model = np.reshape(x_model.dot(c), y.shape)
error = np.sqrt(np.mean((y_model - y)**2))
grad_errors = []
grad_errors = np.append(grad_errors, error)

while True:
    #grad = 1./len(x) * np.array([np.sum(np.array([y_model-y]).T * x_model, axis=0)]).T
    #c = c - alfa * grad
    for t in range(len(x)):
        grad = 1./len(x) * np.array([(y_model[t]-y[t]) * x_model[t,:]]).T
        c = c - alfa * grad


    y_model = np.reshape(x_model.dot(c), y.shape)
    error = np.sqrt(np.mean((y_model - y)**2))
    grad_errors = np.append(grad_errors, error)
    print str(round(error,5))


