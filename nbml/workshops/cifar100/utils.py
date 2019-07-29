from ...tools import *
from keras.datasets import cifar100
from tensorflow import keras
from torch.utils.data import DataLoader, TensorDataset
import torch

vocab = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]

def getCIFAR100():
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
    x_train, x_test = x_train/255, x_test/255
    shapes(x_train, x_test, y_train, y_test)
    return x_train, x_test, y_train, y_test
def torchCIFAR100(x_train, x_test, y_train, y_test, bs=128):
    x_train, x_test = torch.Tensor(x_train).transpose(-1,1), torch.Tensor(x_test).transpose(-1,1)
    y_train, y_test = torch.Tensor(y_train).squeeze(), torch.Tensor(y_test).squeeze()
    shapes(x_train, x_test, y_train, y_test)
    tdl = DataLoader(TensorDataset(x_train, y_train), batch_size=bs, shuffle=True)
    vdl = DataLoader(TensorDataset(x_test, y_test), batch_size=bs, shuffle=True)
    return tdl, vdl