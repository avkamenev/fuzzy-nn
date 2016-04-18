from numpy import linalg as LA
from itertools import product
import scipy


mfs = [[0,0.5], [0.2,0.5], [0.4,0.5], [0.6,0.5], [0.8,0.5], [1,0.5]]

fRules = list(product(range(len(mfs)), range(len(mfs)), range(len(mfs)), range(len(mfs))))
c = np.random.rand((x.shape[1]+1)*len(fRules),1)
#c = np.random.rand(number_of_rules, x.shape[1]+1)
#c = np.zeros(((x.shape[1]+1)*len(fRules),1))

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

for loop_numbers in range(100):
    if loop_numbers % 10 == 0:
        print loop_numbers
    for t in range(len(x)):
        #y_from_rule = x_with_one[t,:].dot(c.T)
        #y_model = np.sum(beta[t,:] * y_from_rule)

        x_hat = np.reshape(np.array([x_with_one[t,:]]).T.dot(np.array([beta[t,:]])), ((x.shape[1]+1)*len(fRules),1))
        y_model = c.T.dot(x_hat)
        c = c + 0.1*( (y[t] - y_model) / LA.norm(x_hat) ) * x_hat

x_hat = np.zeros((len(x), (x.shape[1]+1)*len(fRules)))
for i in range(len(x)):
    x_hat[i,:] = np.reshape(np.array([x_with_one[t,:]]).T.dot(np.array([beta[t,:]])), ( (x.shape[1]+1)*len(fRules)))
y_model = x_hat.dot(c)
print round(np.sum(abs(y_model - y)/abs(y))/len(y),2)





c = np.zeros(((x.shape[1]+1)*len(fRules),1))
#Q = np.zeros((c.shape[0]*c.shape[1], c.shape[0]*c.shape[1]))
Q = 100*np.identity(c.shape[0]*c.shape[1])
for loop_numbers in range(1):
    for t in range(len(x)):
        #y_from_rule = x_with_one[t,:].dot(c.T)
        #y_model = np.sum(beta[t,:] * y_from_rule)
        #x_hat = np.array([beta[t,:]]).T.dot(np.array([x_with_one[t,:]]))
        #x_hat = np.reshape(x_hat, (len(Q),1))

        x_hat = np.reshape(np.array([x_with_one[t,:]]).T.dot(np.array([beta[t,:]])), ((x.shape[1]+1)*len(fRules),1))
        y_model = c.T.dot(x_hat)

        Q = Q - ( Q.dot(x_hat).dot(x_hat.T).dot(Q) )/( 1 + x_hat.T.dot(Q).dot(x_hat) )
        c = c + Q.dot(x_hat) * (y[t] - y_model)
        #print get_y_model(c,x,beta)


x_hat = np.zeros((len(x), (x.shape[1]+1)*len(fRules)))
for i in range(len(x)):
    x_hat[i,:] = np.reshape(np.array([x_with_one[t,:]]).T.dot(np.array([beta[t,:]])), ( (x.shape[1]+1)*len(fRules)))
y_model = x_hat.dot(c)
print round(np.sum(abs(y_model - y)/abs(y))/len(y),2)

