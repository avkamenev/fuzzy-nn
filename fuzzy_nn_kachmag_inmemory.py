from numpy import linalg as LA
from itertools import product
from scipy.optimize import fmin
import scipy


#number_of_separations = 3
#step_for_mf = 1./(number_of_separations-1)
#mfs = [[0,0.05], [0.1,0.05], [0.2,0.05], [0.3,0.05], [0.4,0.05], [0.5,0.05], [0.6,0.05], [0.7,0.05], [0.8,0.05], [0.9,0.05], [1,0.05]]
#mfs = [[0,0.05], [0.2,0.05], [0.4,0.05], [0.6,0.05], [0.8,0.05], [1,0.05]]
#mfs = [[0,0.05], [0.5,0.05], [1,0.05]]
mfs = [[0,0.3], [1,0.3]]

#fRules = list(product(range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs))))
fRules = list(product(range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs))))

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

np.random.seed(1234)
c = (np.random.rand((x.shape[1]+1)*len(fRules),1)-0.5) * 2
#c = np.random.rand(number_of_rules, x.shape[1]+1)
#c = np.zeros(((x.shape[1]+1)*len(fRules),1))

old_error=10
error=2
loop_numbers=0

y_model = np.reshape(x_model.dot(c), y.shape)
error = np.sqrt(np.mean((y_model - y)**2))
kachmag_errors = []
kachmag_errors = np.append(kachmag_errors, error)
print 'prepared'

while ((old_error-error)>=0.00001) & (error>0.03):
#while loop_numbers!=100:
    loop_numbers += 1

    for t in range(len(x)):
        y_model = x_model[t,:].dot(c)
        c = c + 0.1*( (y[t] - y_model) / LA.norm(x_model[t,:])**2 ) * np.reshape(x_model[t,:], c.shape)


    old_error = error
    y_model = np.reshape(x_model.dot(c), y.shape)
    error = np.sqrt(np.mean((y_model - y)**2))
    kachmag_errors = np.append(kachmag_errors, error)
    print str(loop_numbers) + ':'
    print str(round(error,5)) + ' with improvement: ' + str(round(old_error-error,5))
#    y_test_model = np.reshape(x_test_model.dot(c), y_test.shape)
#    print 'TEST:'+str(round(np.sqrt(np.mean((y_test_model - y_test)**2)),5))


print np.sqrt(np.mean((y_model - y)**2))
print np.sqrt(np.mean((y_test_model - y_test)**2))



def f(mf_params):
    #mfs = [[mf_params[0],mf_params[1]], [mf_params[2],mf_params[3]], [mf_params[4],mf_params[5]]]
    #mfs = [[mf_params[0],mf_params[1]], [mf_params[2],mf_params[3]], [mf_params[4],mf_params[5]], [mf_params[6],mf_params[7]]]
    mfs = [[mf_params[0],mf_params[1]], [mf_params[2],mf_params[3]], [mf_params[4],mf_params[5]], [mf_params[6],mf_params[7]], [mf_params[8],mf_params[9]], [mf_params[10],mf_params[11]]]
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

    y_model = np.reshape(x_model.dot(c), y.shape)
    error = np.sum(abs(y_model - y)/abs(y))/len(y)

    return error

mf_params = [0,0.05, step_for_mf,0.05, 1,0.05]
mf_params = [0,0.05, 0.2,0.05, 0.4,0.05, 0.6,0.05, 0.8,0.05, 1,0.05]
xopt = fmin(f, mf_params, xtol=0.01, ftol=0.01, maxiter=50)
print f(xopt)
mfs = [[xopt[0],xopt[1]], [xopt[2],xopt[3]], [xopt[4],xopt[5]], [xopt[6],xopt[7]], [xopt[8],xopt[9]], [xopt[10],xopt[11]]]
#mfs = [[xopt[0],xopt[1]], [xopt[2],xopt[3]], [xopt[4],xopt[5]]]

