from numpy import linalg as LA
number_of_rules = 2**x.shape[1]
number_of_mfs = 2
mfs = [[0,1],[1,1]]
c = np.random.rand(number_of_rules, x.shape[1]+1)
#c = np.zeros((number_of_rules, x.shape[1]+1))

fRules = list(product([0, 1], [0, 1], [0, 1], [0, 1]))

w_values = np.zeros((len(x), number_of_rules))
for n in range(len(fRules)):
    rule = fRules[n]
    w_value = 0
    for i in range(len(rule)):
        w_value = w_value + scipy.stats.norm(mfs[rule[i]][0], mfs[rule[i]][1]).pdf(x[:,i])
    w_values[:,n] = w_value

beta = w_values/np.array([np.sum(w_values, axis=1)]).T
x_with_one = np.column_stack((np.ones(len(x)).T, x))

for loop_numbers in range(1):
    for t in range(len(x)):
        y_from_rule = x_with_one[t,:].dot(c.T)
        y_model = np.sum(beta[t,:] * y_from_rule)

        x_hat = np.array([beta[t,:]]).T.dot(np.array([x_with_one[t,:]]))
        #x_hat = x[t,:]
        c = c + ( (y[t] - y_model) / LA.norm(x_hat) ) * x_hat


        print get_y_model(c,x)





Q = np.zeros((c.shape[0]*c.shape[1], c.shape[0]*c.shape[1]))
for t in range(len(x)):
    y_from_rule = x_with_one[t,:].dot(c.T)
    y_model = np.sum(beta[t,:] * y_from_rule)
    x_hat = np.array([beta[t,:]]).T.dot(np.array([x_with_one[t,:]]))
    x_hat = np.reshape(x_hat, (len(Q),1))

    Q = Q - ( Q.dot(x_hat).dot(x_hat.T).dot(Q) )/( 1 + x_hat.T.dot(Q).dot(x_hat) )
    c = c + np.reshape(Q.dot(x_hat) * (y[t] - np.reshape(c, ((len(Q),1))).T.dot(x_hat)), (c.shape))
    print get_y_model(c,x)

