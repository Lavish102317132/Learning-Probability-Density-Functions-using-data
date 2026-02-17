import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

roll = int(input())

ar = 0.5*(roll%7)
br = 0.3*((roll%5)+1)

data = pd.read_csv("india_no2.csv")

x = data["NO2"].dropna().values

z = x + ar*np.sin(br*x)

z = (z - z.mean())/z.std()

z = torch.tensor(z,dtype=torch.float32).view(-1,1)

class G(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1,16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Linear(16,1)
        )
    def forward(self,x):
        return self.net(x)

class D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1,16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Linear(16,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.net(x)

g = G()
d = D()

optg = optim.Adam(g.parameters(),lr=0.001)
optd = optim.Adam(d.parameters(),lr=0.001)

loss = nn.BCELoss()

epochs = 2000
batch = 128

for i in range(epochs):

    idx = np.random.randint(0,len(z),batch)
    real = z[idx]

    noise = torch.randn(batch,1)
    fake = g(noise)

    dreal = d(real)
    dfake = d(fake.detach())

    lreal = loss(dreal,torch.ones_like(dreal))
    lfake = loss(dfake,torch.zeros_like(dfake))

    ld = lreal+lfake

    optd.zero_grad()
    ld.backward()
    optd.step()

    noise = torch.randn(batch,1)
    fake = g(noise)

    dfake = d(fake)

    lg = loss(dfake,torch.ones_like(dfake))

    optg.zero_grad()
    lg.backward()
    optg.step()

noise = torch.randn(5000,1)
gen = g(noise).detach().numpy().flatten()

kde = gaussian_kde(gen)

xs = np.linspace(gen.min(),gen.max(),500)
ys = kde(xs)

plt.hist(gen,bins=50,density=True,alpha=0.5)
plt.plot(xs,ys)
plt.savefig("pdf.png")
plt.show()

print(ar,br)
