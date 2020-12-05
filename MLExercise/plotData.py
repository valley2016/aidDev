import numpy as np
import matplotlib.pyplot as plt

# load exercise data
train = np.loadtxt('click.csv',delimiter=',',skiprows=1)
train_x = train[:,0]
train_y = train[:,1]

# show
plt.plot(train_x,train_y,'o')
plt.show()

