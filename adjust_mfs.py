from scipy.optimize import fmin

def f(mf_params):
    mfs_int = np.reshape(mf_params, (len(mfs), 2))


    np.random.seed(1234)
    y_model = np.zeros(len(x))

    x_mf_internal = np.zeros((len(mfs_int),x.shape[0],x.shape[1]))
    for i in range(len(mfs_int)):
        x_mf_internal[i] = scipy.stats.norm(mfs_int[i][0], mfs_int[i][1]).pdf(x)

    for t in range(len(x)):
        w_values = x_mf_internal[np.array(fRules)[:,range(x.shape[1])], t, range(x.shape[1])]
        w_values = np.max(w_values, axis=1)
        beta_t = w_values/np.sum(w_values)
        x_model_t = np.reshape(np.array([x_with_one[t,:]]).T.dot(np.array([beta_t])), ((x.shape[1]+1)*len(fRules)))
        y_model[t] = x_model_t.dot(c)

    error = np.sqrt(np.mean((y_model - y)**2))
    print 'ERROR: ' + str(error)
    return error

mf_params = np.reshape(mfs, (len(mfs)*2))
xopt = fmin(f, mf_params, xtol=0.01, ftol=0.01, maxiter=3)
print f(xopt)

mfs = np.reshape(xopt, (len(mfs), 2))