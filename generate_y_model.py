import scipy
from itertools import product

def get_y_model(c,x,beta, fRules):
    number_of_rules = 2**x.shape[1]
    number_of_mfs = 2
    mfs = [[0,1],[0.5,1],[1,1]]

    fRules = list(product([0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]))

    #w_values = np.zeros((len(x), number_of_rules))
    #for n in range(len(fRules)):
    #    rule = fRules[n]
    #    w_value = 0
    #    for i in range(len(rule)):
    #        w_value = w_value + scipy.stats.norm(mfs[rule[i]][0], mfs[rule[i]][1]).pdf(x[:,i])
    #    w_values[:,n] = w_value

    #beta = w_values/np.array([np.sum(w_values, axis=1)]).T
    x_with_one = np.column_stack((np.ones(len(x)).T, x))
    #y_from_rules = x_with_one.dot(c.T)
    #y_model = np.sum(beta * y_from_rules, axis=1)

    x_hat = np.zeros((len(x), (x.shape[1]+1)*len(fRules)))
    for i in range(len(x)):
        x_hat[i,:] = np.reshape(np.array([x_with_one[t,:]]).T.dot(np.array([beta[t,:]])), ( (x.shape[1]+1)*len(fRules)))

    y_model = x_hat.dot(c)

    return round(np.sum(abs(y_model - y)/abs(y))/len(y),2)
