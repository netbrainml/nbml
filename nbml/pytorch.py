import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from tools import *

def cc(x, y=None):
    if torch.cuda.device_count(): return x.cuda() if not y else x.cuda().y
    return x if not y else x.y

class BasicTrainableClassifier(nn.Module):
    def __init__(self, crit=nn.CrossEntropyLoss(), rg = True):
        super().__init__()
        self.crit = crit
        try:
            self.crit(torch.ones(1,1), torch.ones(1,1).long())
            self.dtype = long()
        except:
            self.dtype = float()
        self.rg = rg 
        self.train_acc = []; self.valid_acc = []
        self.train_loss = []; self.valid_loss = []
    
    def fit(self, train_ds, valid_ds, cbs=False,
            epochs=1, learning_rate=1e-3):
        self.train()
        op = torch.optim.Adam(self.parameters(), lr=learning_rate)
        for e in range(epochs):
            print(f"Epoch {e+1}")
            for data in tqdm(train_ds):
                op.zero_grad()
                pred = self(cc(data[0]))
                loss = self.crit(pred, cc(data[1], self.dtype))
                loss.backward(retain_graph = self.rg)
                op.step()
            self.cbs_(train_ds, valid_ds=valid_ds) if cbs else None
            
    def acc(self, out, Y):
        return (torch.argmax(out, dim=1)==Y.long()).float().mean()
    
    def cbs_(self, train_ds, valid_ds=None):
        self.eval()
        train_acc, train_loss = 0,0
        valid_acc, valid_loss = 0,0
        for idx,data in enumerate(train_ds):
            train_pred = self(cc(data[0]))
            train_acc += self.acc(train_pred, cc(data[1], self.dtype)).item()
            train_loss += self.crit(train_pred, cc(data[1], self.dtype)).item()
        for idx,data in enumerate(valid_ds):
            valid_pred = self(cc(data[0]))
            valid_acc += self.acc(valid_pred, cc(data[1], self.dtype)).item()
            valid_loss += self.crit(valid_pred, cc(data[1], self.dtype)).item()

        self.train_acc.append(train_acc/(idx+1))
        self.train_loss.append(train_loss/(idx+1))
        self.valid_acc.append(valid_acc/(idx+1))
        self.valid_loss.append(valid_loss/(idx+1))
        print(f"Accuracy: (V:{self.valid_acc[-1]}, T:{self.train_acc[-1]}), Loss: (V:{self.valid_loss[-1]}, T:{self.train_loss[-1]})")
        self.train()

    @property
    def plot(self):
        fig = plt.figure(figsize=(15,7.5), dpi= 80)
        plt.subplot(1, 2, 1)
        plen(self.valid_acc, label_ = "validation"), plen(self.train_acc, label_ = "training")
        plt.legend()
        plt.title(f"Accuracy: (V:{self.valid_acc[-1]}, T:{self.train_acc[-1]})")
        plt.subplot(1, 2, 2)
        plen(self.valid_loss, label_ = "validation"), plen(self.train_loss, label_ = "training")
        plt.legend()
        plt.title(f"Loss: (V:{self.valid_loss[-1]}, T:{self.train_loss[-1]})")
        plt.show()