import numpy as np
import matplotlib.pyplot as plt

# data standardize
def standardize(data):
    data_mean = data.mean()
    data_sigma = data.std()
    return (data - data_mean) / data_sigma

# predict function
def predictFunc(x):
    return theta0 + theta1 * x

# cost function
def costFunc(x, y):
    return 0.5 * np.sum((y - predictFunc(x))**2)

# load exercise data
train = np.loadtxt('click.csv',delimiter=',',skiprows=1)
train_x = train[:,0]
train_y = train[:,1]
train_z = standardize(train_x)

# parameter init
theta0 = np.random.rand()
theta1 = np.random.rand()

learningRate = 1e-3  # 0.001
errorDiff = 1
cnt = 0

plt.ion()    # set interactive mode

error = costFunc(train_z,train_y)
while(errorDiff > 1e-2):
    plt.clf() 
    plt.suptitle("Linear Func Dynamic Disney",fontsize=15) 

    # calculate parameter
    newTheta0 = theta0 - learningRate * np.sum(predictFunc(train_z) - train_y)
    newTheta1 = theta1 - learningRate * np.sum((predictFunc(train_z) - train_y) * train_z)
    # updata parameter
    theta0 = newTheta0
    theta1 = newTheta1
    # updata error & errorDiff
    curError = costFunc(train_z,train_y)
    errorDiff = error - curError
    error = curError
    cnt += 1
    # print log
    log = 'cnt = {}: theta0 = {:.3f}, theta1 = {:.3f}, diff = {:.4f}'
    print(log.format(cnt,theta0,theta1,errorDiff))
    # set axis range
    plt.xlim((-2.5, 2.5))
    plt.ylim((100, 750))
    # show
    x = np.linspace(-2, 2, 100)
    plt.plot(train_z, train_y, 'o')
    plt.plot(x, predictFunc(x))
    plt.pause(0.02)

print("Learning is Completed！")
plt.ioff()       # exit interactive mode
plt.show() 
