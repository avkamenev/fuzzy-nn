
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
#print round(np.sum(abs(y_test_model - y_test)/abs(y_test))/len(y_test),5)

print np.sqrt(np.mean((y_test_model - y_test)**2))

