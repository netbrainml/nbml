import pandas as pd
import cv2
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from ...pytorch import *


def get_data(path=None, full=False, top=10, **_):
    df = pd.read_csv(path+"labels.csv")
    if full: return df['id'], df['breed']
    top_breeds = sorted(list(df['breed'].value_counts().head(top).index))
    df_top = df[df['breed'].isin(top_breeds)]
    return df_top['id'], df_top['breed']

def get_numpy(imloc, label, path=None, dim=224, dir_="train", **_):
    x, y = [],[]
    classes = list(set(label))
    nc = len(classes)
    for data in tqdm(zip(imloc,label),total=len(imloc)):
        x.append(cv2.resize(cv2.imread(path+f"{dir_}/{data[0]}.jpg"), (dim,dim)))
        y.append(classes.index(data[1]))
    return np.array(x), np.array(y)

def nptotorch(x,y, split=0.6, subset=None, verbose=True, nc=120, bs=256, **_):
    x_t = torch.tensor(x).transpose(-1,1)
    y_t = torch.tensor(y)
    idx = get_idx(x_t)[:subset] if subset is not None else get_idx(x_t)
    loc = int(split * subset) if subset is not None else int(split * idx.shape[0])
    tdl = DataLoader(TensorDataset(x_t[idx[:loc]].float()
                               ,y_t[idx[:loc]]
                              ), batch_size = bs)
    vdl = DataLoader(TensorDataset(x_t[idx[loc:]].float()
                                   ,y_t[idx[loc:]]
                                  ), batch_size = bs)
    return tdl,vdl

def load_data(**kwargs):
    return nptotorch(*get_numpy(*get_data(**kwargs), **kwargs), **kwargs)