from numpy import linalg as LA
from itertools import product
import scipy


mfs = [[0,0.1], [0.5,0.1], [1,0.1]]

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


#c = np.zeros(((x.shape[1]+1)*len(fRules),1))
c = np.random.rand((x.shape[1]+1)*len(fRules),1)-0.5
#Q = np.zeros((c.shape[0]*c.shape[1], c.shape[0]*c.shape[1]))
Q = 100*np.identity(c.shape[0]*c.shape[1])
for loop_numbers in range(10):
    if loop_numbers % 1 == 0:
        print str(loop_numbers) + ':'
        y_model = np.reshape(x_model.dot(c), y.shape)
        print round(np.sum(abs(y_model - y)/abs(y))/len(y),3)
    for t in range(len(x)):
        #y_from_rule = x_with_one[t,:].dot(c.T)
        #y_model = np.sum(beta[t,:] * y_from_rule)
        #x_hat = np.array([beta[t,:]]).T.dot(np.array([x_with_one[t,:]]))
        #x_hat = np.reshape(x_hat, (len(Q),1))

        #x_hat = np.reshape(np.array([x_with_one[t,:]]).T.dot(np.array([beta[t,:]])), ((x.shape[1]+1)*len(fRules),1))
        y_model = c.T.dot(x_model[t,:])
        x_hat = np.reshape(x_model[t,:], c.shape)

        Q = Q - ( Q.dot(x_hat).dot(x_hat.T).dot(Q) )/( 1 + x_hat.T.dot(Q).dot(x_hat) )
        c = c + Q.dot(x_hat) * (y[t] - y_model)
        #print get_y_model(c,x,beta)


y_model = np.reshape(x_model.dot(c), y.shape)
print round(np.sum(abs(y_model - y)/abs(y))/len(y),3)

