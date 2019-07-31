from ...pytorch import *
import cv2 as cv
import pickle

class rC3D(BasicTrainableClassifier):
    def __init__(self, in_c, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.ls = nn.Sequential(ConvRelu(in_c, 16, 3, 1, 1),
                                rC3DBlock(16, 32),
                                rC3DBlock(32, 64),
                                rC3DBlock(64, 128),
                                rC3DBlockMP(128, 128, True))
        self.pool = StackPool(10)
        self.fc = nn.Linear(2*128*10**3,num_classes)
        init_cnn(self)
    def __call__(self, X):
        fm = self.ls(X)
        pfm = self.pool(fm)
        ft = self.fc(torch.flatten(pfm,start_dim=1))
        return ft

class grC3D(BasicTrainableClassifier):
    def __init__(self, in_c, num_classes, path, **kwargs):
        super().__init__(**kwargs)
        with open(path, "rb") as mfile:
            self.gan = pickle.load(mfile)
        self.model = rC3D(in_c, num_classes)
        init_cnn(self)
    def __call__(self, x):
        gX = torch.cat([self.gan.generator.forward(img)[None,:] for img in x], dim=0) 
        return self.model(gX)

class OFrC3D(BasicTrainableClassifier):
    def __init__(self, in_c, num_classes,**kwargs):
        super().__init__(**kwargs)
        self.model = rC3D(in_c, num_classes)
        
    def __call__(self,x):
        out = None
        with torch.no_grad(): 
            for b in x:
                first = self.tonp(b[0])
                second = self.tonp(b[-1])
                ro = self.denseOF(first, second)
                out = self.tot(ro)[None,None,] if out is None else torch.cat((out,self.tot(ro)[None,None,]),
                                                                        dim=0)
        return self.model(out)
    
    def denseOF(self, first, second):
        prev = cv.cvtColor(first, cv.COLOR_BGR2GRAY)
        mask = np.zeros_like(first)
        mask[..., 1] = 255
        last = cv.cvtColor(second, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prev, last,
                                           None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
        mask[..., 0] = angle * 180 / np.pi / 2
        mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
        return cv.cvtColor(mask, cv.COLOR_HSV2BGR)
    
    def tonp(self,x): return x.transpose(0,2).cpu().numpy()
    def tot(self,x): return torch.Tensor(x).transpose(0,2).cuda().data.detach()

def init_cnn(m):
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv3d,nn.Conv2d,nn.Linear)): nn.init.kaiming_normal_(m.weight)
    for l in m.children(): init_cnn(l)
        
def ResConv(in_c, kernel_size, stride):
    pad = kernel_size//2
    conv1 = nn.Conv3d(in_c, in_c//2, 1, stride)
    conv2 = nn.Conv3d(in_c//2, in_c//2, kernel_size, stride, padding=pad)
    conv3 = nn.Conv3d(in_c//2, in_c, kernel_size, stride, padding=pad)
    return nn.Sequential(conv1, conv2, conv3)

def ConvRelu(in_c,out_c,ks,stride,padding):
    conv1 = nn.Conv3d(in_c, out_c, ks, stride,padding)
    return nn.Sequential(conv1,nn.LeakyReLU())

class rC3DBlock(BasicTrainableClassifier):
    def __init__(self, in_c, out_c, ks=3,**kwargs):
        super().__init__(**kwargs)
        self.rconv1 = ResConv(in_c, ks, ks//2); self.bn1 = nn.BatchNorm3d(in_c)
        self.conv1 = ConvRelu(in_c, out_c, ks, 1, ks//2)
    def __call__(self, X):
        fm1 = self.bn1(F.leaky_relu(self.rconv1(X) + X))
        cfm1 = self.conv1(fm1)
        return cfm1

class rC3DBlockMP(BasicTrainableClassifier):
    def __init__(self, in_c, out_c, pad=False,**kwargs):
        super().__init__(**kwargs)
        self.rconv1 = ResConv(in_c, 3, 1); self.conv1 = ConvRelu(in_c, out_c, 3, 1, 1)
        self.bn1 = nn.BatchNorm3d(in_c)
        self.mp1 = nn.MaxPool2d((2,2), 2, 1) if pad else nn.MaxPool2d((2,2), 2)
    def __call__(self, X):
        fm1 = self.bn1(F.leaky_relu(self.rconv1(X) + X))
        init = True
        for img in fm1:
            pfm1 = self.mp1(img).unsqueeze(0) if init else torch.cat([pfm1, self.mp1(img).unsqueeze(0)], dim = 0)
            init = False
        cfm1 = F.relu(self.conv1(pfm1)).sub_(0.4)
        return cfm1

class StackPool(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.max = nn.AdaptiveMaxPool3d(size)
        self.avg = nn.AdaptiveAvgPool3d(size)
    def __call__(self, X):
        return torch.cat((self.max(X), self.avg(X)), dim=1)


def ConvRelu(in_c,out_c,ks,stride,padding=0):
    conv1 = nn.Conv3d(in_c, out_c, ks, stride, padding)
    return nn.Sequential(conv1,nn.ReLU())

class C3DBlock(BasicTrainableClassifier):
    def __init__(self,nc,nf,padding=0):
        super().__init__()
        self.block = nn.Sequential(ConvRelu(nc,nf,3,1,padding),
                                   ConvRelu(nf,nf,3,1,padding),
                                   nn.MaxPool3d(2,2, padding, ceil_mode=True))
    def forward(self,x): return self.block(x)

class C3D(BasicTrainableClassifier):
    def __init__(self, nc, no):
        super().__init__()
        self.cp_model = nn.Sequential(ConvRelu(nc, 64, 3, 1, 1),
                                      nn.MaxPool3d((1,2,2),(1,2,2)),
                                      ConvRelu(64, 128, 3, 1, 1),
                                      nn.MaxPool3d((2,2,2),2,(1,0,0)),
                                      C3DBlock(128,256,padding=(1,0,0)),
                                      C3DBlock(256,512,padding=1),
                                      C3DBlock(512,512,padding=(1,0,0)),
                                      C3DBlock(512,512,padding=1))
        self.fc = nn.Sequential(nn.Linear(4096,4096),
                                nn.Linear(4096,no))
    def forward(self, x):
        return self.fc(torch.flatten(self.cp_model(x), start_dim=1))