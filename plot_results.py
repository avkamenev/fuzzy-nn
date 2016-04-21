import matplotlib.pyplot as plt

interval=50
labels = ['grad alfa=300', 'grad alfa=100', 'grad alfa=10', 'Kachmag alfa=0.1']


plt.plot(range(len(kachmag_errors[:interval])), kachmag_errors[:interval], label=labels[3], linewidth=2)
plt.plot(range(len(grad_errors_300[:interval])), grad_errors_300[:interval], label=labels[0], linewidth=2)
plt.plot(range(len(grad_errors_100[:interval])), grad_errors_100[:interval], label=labels[1], linewidth=2)
plt.plot(range(len(grad_errors_10[:interval])), grad_errors_10[:interval], label=labels[2], linewidth=2)
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.show()

#