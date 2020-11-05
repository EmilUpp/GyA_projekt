import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

initialData = []

while len(initialData) < 2000:
    nextData = torch.tensor([float(random.randint(0,1)), float(random.randint(0,1)), float(random.randint(0,1)), float(random.randint(0,1))])
    isEven = torch.tensor([float((nextData[0] + nextData[1] * 2 + nextData[2] * 4 + nextData[3] * 8) % 2 == 0)])
    initialData.append([nextData, isEven])


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)
        self.fc3 = nn.Linear(2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for data in initialData:
    input, answer = data

    optimizer.zero_grad()

    output = net(input)
    loss = criterion(output, answer)
    loss.backward()
    optimizer.step()

    print(output, answer)