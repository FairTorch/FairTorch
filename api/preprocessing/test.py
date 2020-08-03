from map import FairMap
import torch


X = torch.randn(20, 10)
X[:10] = 10 * X[:10]
protected = torch.zeros(20, dtype=torch.long)
protected[:10] = 0
protected[10:] = 1

X = (X - torch.mean(X)) / torch.std(X)

fm = FairMap(10, 2, 1, 10, lr=0.01, eta=0.4)
fm.fit(X, protected, 1000)
