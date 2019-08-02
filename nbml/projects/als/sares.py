from ...pytorch import *
from fastai.torch_core import Module
from fastai.layers import conv_layer
from functools import partial

def init_cnn(m):
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv3d,nn.Conv2d,nn.Linear)): nn.init.kaiming_normal_(m.weight)
    for l in m.children(): init_cnn(l)

class noop(Module):
    def __call__(self, x): return x
class StackPool(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.max = nn.AdaptiveMaxPool3d(size)
        self.avg = nn.AdaptiveAvgPool3d(size)
    def __call__(self, X):
        return torch.cat((self.max(X), self.avg(X)), dim=1)
class ResBlockA(Module):
    def __call__(self, x): return self.bn(self.act(self.layer(x)+self.idlayer(x)))
class ResBlockC(Module):
    def __call__(self, x): return self.act(self.op((torch.cat([self.layer(x),self.idlayer(x)],dim=2))))

def P3Dconv(ni,nf,stride=1,padding=1, s=True, t=False):
    return nn.Conv3d(ni,nf,kernel_size=(3,1,1) if not s or t else (1,3,3),stride=1,
                     padding=padding,bias=False)

saconv2d = partial(conv_layer, self_attention=True)

class SAconv3d(Module):
    def __init__(self, nc, nf, stride=1, padding=1):
        self.saconv = saconv2d(nc,nf)
    def __call__(self, x):
        return torch.cat([self.saconv(img.squeeze()).transpose(1,0)[None,:] for img in x])

class aP3Da(ResBlockC):
    def __init__(self, ni, nf, nc,**kwargs):
        super().__init__()
        self.layer = nn.Sequential(P3Dconv(nf,nf,padding=0, s=True),
                                   P3Dconv(nf,nf,padding=1, t=True))
        self.idlayer = SAconv3d(nc, nf)
        self.op = nn.Conv3d(nf, nf, 3, **kwargs)
        self.act = nn.ReLU(inplace=True)

class P3Da(ResBlockA):
    def __init__(self, ni, nf, pad=True,**kwargs):
        super().__init__()
        self.layer = nn.Sequential(nn.Conv3d(ni,nf,1),
                                   P3Dconv(nf,nf,padding=(0,1,1) if pad else 0, s=True),
                                   P3Dconv(nf,nf,padding=(1,0,0) if pad else 0, t=True),
                                   nn.Conv3d(nf,nf,1),)
        self.idlayer = nn.Conv3d(ni, nf, 3, **kwargs)
        self.act = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm3d(nf)

class FlattenDim(Module):
    def __init__(self, dim): self.dim=dim
    def __call__(self,x): return torch.flatten(x, start_dim=self.dim)

class P3DaModel(BasicTrainableClassifier):
    def __init__(self, ni, no, **kwargs):
        super().__init__(**kwargs)
        self.model = nn.Sequential(P3Da(ni, 64, padding=1),
                                   nn.MaxPool3d((1,2,2),(1,2,2)),
                                   P3Da(64, 128, padding=1),
                                   nn.MaxPool3d(2,2, padding=(1,0,0)),
                                   P3Da(128, 256, padding=1),
                                   P3Da(256, 512, padding=1),
                                   nn.MaxPool3d(2,2),
                                   P3Da(512, 512, padding=1),
                                   P3Da(512, 512, padding=1),
                                   nn.MaxPool3d(2,2),
                                   P3Da(512, 512, padding=1),
                                   P3Da(512, 512, padding=1),
                                   nn.MaxPool3d(2,2),
                                   FlattenDim(1),
                                   nn.Linear(8192,4096),
                                   nn.Linear(4096,no)
                                  )
        init_cnn(self)
    def __call__(self,x): return self.model(x)