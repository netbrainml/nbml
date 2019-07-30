import random
from ...tools import *
import time
from IPython.display import clear_output

def plotlo(epochs,losses,X,Y,pred):
    fig = plt.figure(figsize=(15,7.5), dpi= 80)
    plt.subplot(1, 2, 1)
    plt.plot(list(range(epochs)),losses)
    plt.title("Losses")
    plt.subplot(1, 2, 2)
    plt.plot(X,np.array(pred).reshape(-1,1), label = "prediction")
    plt.plot(X,Y, label = "actual")
    plt.legend()
    plt.title(f"Output")
    plt.show()

def get_data(size=500, verbose=True, neg=False):
    X = np.arange(-size,size).reshape(-1,1) if neg else np.arange(size).reshape(-1,1)
    slope = random.randint(2,50); bias = random.randint(2,50)
    Y = np.array(X * slope + bias)
    if verbose: print(f"Our actual slope and bias is: {slope, bias}")
    return X,Y, slope, bias

def visualize(X,Y,pred, rt=False)
    plt.plot(X,Y, label = "Actual")
    plt.plot(X,pred, label="Prediction")
    plt.legend()
    plt.show()

    if rt:
        time.sleep(0.01)
        plt.close()
        clear_output(wait=True)

def MSELoss(X, Y_actual, Y_pred):
    return 1/(2*X.shape[0]) * np.sum(
                                np.square(
                                    np.subtract(Y_pred,Y_actual)
                                ),axis=0)

def dMSELoss(X,Y_actual, Y_pred, parameters):
    dW = 1/X.shape[0] * np.multiply(np.sum(np.subtract(Y_pred,Y_actual)),parameters[0])
    dB = 1/X.shape[0] * np.sum(np.subtract(Y_pred,Y_actual))
    return dW, dB

def gradient_descent(X,Y_actual, Y_pred, parameters, learning_rate = 0.01):
    dW, dB =dMSELoss(X,Y_actual, Y_pred, parameters)
    parameters[0] -= np.multiply(learning_rate,dW)
    parameters[1] -= np.multiply(learning_rate,dB)