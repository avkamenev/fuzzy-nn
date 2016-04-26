
###############
# TEST BY ROW #
###############
x_test_mf = np.zeros((len(mfs),x_test.shape[0],x_test.shape[1]))
for i in range(len(mfs)):
    x_test_mf[i] = scipy.stats.norm(mfs[i][0], mfs[i][1]).pdf(x_test)

x_test_with_one = np.column_stack((np.ones(len(x_test)).T, x_test))
y_test_model = np.zeros(len(x_test))
for t in range(len(x_test)):
    if t%1000==0:
        print t
    w_values = x_test_mf[np.array(fRules)[:,range(x_test.shape[1])], t, range(x_test.shape[1])]
    w_values = np.max(w_values, axis=1)
    beta_t = w_values/np.sum(w_values)
    x_model_t = np.reshape(np.array([x_test_with_one[t,:]]).T.dot(np.array([beta_t])), ((x_test.shape[1]+1)*len(fRules)))
    y_test_model[t] = x_model_t.dot(c)


error = np.sqrt(np.mean((y_test_model - y_test)**2))
print error






res = pd.read_csv('fuzy_nn_result_0_466277.csv', index_col='id')
out = res
res['relevance'][res['relevance']>3] = 3
min(res['relevance'])

out = pd.DataFrame({'id': res.index, 'relevance': res.relevance})
out.to_csv('fuzy_nn_result_0_466277.csv', index=None)
