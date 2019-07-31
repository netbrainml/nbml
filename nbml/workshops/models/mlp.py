from ...pytorch import *
class MLP(BasicTrainableClassifier):
    def __init__(self, ls, act = nn.ReLU()):
        super().__init__()
        self.model = nn.Sequential(*[nn.Sequential(nn.Linear(*n), act) for n in ls])
    def forward(self, X): return self.model(X)
    