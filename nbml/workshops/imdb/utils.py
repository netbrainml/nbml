from ...pytorch import *
from keras.preprocessing import sequence
from keras.datasets import imdb
from torch.utils.data import DataLoader, TensorDataset

def sliceton(X,Y,n,dictv):
    out = []
    ys = np.array([])
    for i,review in enumerate(tqdm(X)):
        for idx in range(0, len(review),n):
            if idx+n>len(review)-1:
                out.append(np.array(sequence.pad_sequences(np.array(review[idx: idx+n])[None,:],
                                                           value=dictv["<PAD>"], maxlen=n,
                                                           padding='post')).squeeze())
                ys = np.append(ys,Y[i])
                break    
            out.append(review[idx: idx+n])
            ys = np.append(ys,Y[i])
    return np.array(out), ys

def ntot(X):
    return torch.stack([torch.Tensor(x) for x in X])[...,None]

def load_imdb():
    old = np.load
    np.load = lambda *a,**k: old(*a,**k,allow_pickle=True)
    return imdb.load_data(num_words=1000)
def load_dict():
    dictv = imdb.get_word_index()
    dictv = {k:(v+3) for k,v in dictv.items()}
    dictv["<PAD>"] = 0
    dictv["<START>"] = 1
    dictv["<UNK>"] = 2
    dictv["<UNUSED>"] = 3
    return dictv
def process_imdb(X_train, y_train, X_test, y_test, n, dictv, bs=64):
    X_train_pds, y_train_pds = sliceton(X_train, y_train, n, dictv)
    X_test_pds, y_test_pds = sliceton(X_test, y_test, n, dictv)
    x_train, x_test = ntot(X_train_pds), ntot(X_test_pds)
    y_train, y_test = torch.Tensor(y_train_pds), torch.Tensor(y_test_pds)
    tdl = DataLoader(TensorDataset(x_train, y_train), batch_size=bs, shuffle=True)
    vdl = DataLoader(TensorDataset(x_test, y_test), batch_size=bs, shuffle=True)
    return tdl, vdl

def decode_review(text, dictv):
    reverse_word_index = dict([(value, key) for (key, value) in dictv.items()])
    return ' '.join([reverse_word_index.get(i, '?') for i in text])