from numpy import linalg as LA
from itertools import product
import scipy

#mfs = [[0,0.05], [0.2,0.05], [0.4,0.05], [0.6,0.05], [0.8,0.05], [1,0.05]]
#mfs = [[0,0.05], [0.5,0.05], [1,0.05]]
mfs = [[0,0.3], [1,0.3]]
# fRules = list(product(range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs)),range(len(mfs)),range(len(mfs)),range(len(mfs)),range(len(mfs)), range(len(mfs))))
fRules = list(product(range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs))))

np.random.seed(1234)
c = (np.random.rand((x.shape[1]+1)*len(fRules),1)-0.5) * 2
x_with_one = np.column_stack((np.ones(len(x)).T, x))
y_model = np.zeros(len(x))

x_mf = np.zeros((len(mfs),x.shape[0],x.shape[1]))
for i in range(len(mfs)):
    x_mf[i] = scipy.stats.norm(mfs[i][0], mfs[i][1]).pdf(x)

loop_numbers=0
max_loops = 10
alfa = 30
errors = np.zeros(max_loops)
while ((old_error-error)>=0.00001) & (error>0.03) & (loop_numbers!=max_loops):
    print 'Loop: '+str(loop_numbers)
    for t in range(len(x)):
        if t%100==0:
            print t
        w_values = x_mf[np.array(fRules)[:,range(x.shape[1])], t, range(x.shape[1])]
        w_values = np.max(w_values, axis=1)
        beta_t = w_values/np.sum(w_values)
        x_model_t = np.reshape(np.array([x_with_one[t,:]]).T.dot(np.array([beta_t])), ((x.shape[1]+1)*len(fRules)))
        y_model[t] = x_model_t.dot(c)

        grad = 1./len(x) * np.array([(y_model[t]-y[t]) * x_model_t]).T
        c = c - alfa * grad

    error = np.sqrt(np.mean((y_model - y)**2))
    errors[loop_numbers] = error
    print 'ERROR: ' + str(error)
    loop_numbers += 1


print errors
