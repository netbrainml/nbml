from ...pytorch import *
from IPython.display import clear_output
import time
import pandas as pd

def show_imgs(imgs,labels,hm=10, dim=28):
	for i in np.random.permutation(len(imgs))[:hm]:
	    plt.imshow(imgs[i].reshape(dim,dim)/255,'gray')
	    plt.title(labels[i])
	    plt.show()
	    time.sleep(0.1)
	    clear_output(wait=True)

def scale(x): return x/255
def split_vals(x,y,n): idxs = np.random.permutation(len(x)); return x[idxs[:n]], x[idxs[n:]], y[idxs[:n]], y[idxs[n:]]

def loadMNIST(path):
    df_mnist = pd.read_csv(path)
    label = np.array(df_mnist["label"])
    imgs = np.array(df_mnist.drop(['label'],axis=1))
    return imgs, label

def getMNIST(path):
    imgs, label = loadMNIST(path)
    split_amt= round(imgs.shape[0] * 0.90)
    x_train, x_test, y_train, y_test = split_vals(scale(imgs), label, split_amt)
    return x_train, x_test, y_train, y_test 

def seeResults(model, x, y):
    dim=28
    plt.figure(figsize=(20,20))
    idxs = np.random.permutation(len(x))
    for i,idx in enumerate(idxs[:25]):
        plt.subplot(5,5,i+1)
        plt.xticks([]), plt.yticks([])
        plt.grid(False)
        plt.imshow(x[idx].reshape(dim,dim),'gray')
        pred = softmax(model.predict(x[idx][None,:])) #Here the model makes a prediction
        plt.xlabel(f"Label:{y[idx]}, Pred:{np.argmax(pred)}")
    plt.show()

def softmax(x): return np.exp(x.squeeze())/sum(np.exp(x.squeeze()))