import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.nn import MSELoss
from torch.optim import Adam

from model_wrappers import FairModel

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.w = torch.nn.Parameter(torch.randn(1, 3))
        self.b = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        return torch.sum(self.w * x, axis=1, keepdims=True) + self.b

    def fit(self, x, y):
        optimizer = Adam(self.parameters(), lr=0.1)
        criterion = MSELoss()

        for step in range(100):
            self.zero_grad()

            y_ = self(x)
            loss = criterion(y, y_)
            loss.backward()

            optimizer.step()
            if step % 10 == 0:
                print(f"Step: {step}, Loss: {loss.item()}")
        

X = torch.randn(20, 3)
X[:10] = 10 * X[:10]
y = torch.sum(torch.Tensor([[1.0, 0.5, 3.14]]) * X, axis=1, keepdims=True) + 1.41

protected = torch.zeros(20, dtype=torch.long)
protected[:10] = 0
protected[10:] = 1

# define and pretrain model
print("=============== PRETRAINING MODEL =================")

model = Model()
model.fit(X, y)

# wrap pretrained model in FairModel adversary and train
print("=============== FITTING FAIRMODEL =================")

fm = FairModel(model, output_size=1, n_groups=2, n_hidden=1, layer_width=10)
fm.fit(X, y, protected, 0.5, steps=1000)
