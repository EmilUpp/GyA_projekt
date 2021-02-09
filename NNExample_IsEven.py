import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

initialData = []

# initialize data as [[list of bits], [isEven]]
while len(initialData) < 2000:  # initialize 2000 examples
    # generate an example
    # the input must be a tensor of floats, therefore an int is represented by a list of bits as either 1.0f or 0.0f
    nextData = torch.tensor([float(random.randint(0, 1)), float(random.randint(0, 1)), float(random.randint(0, 1)),
                             float(random.randint(0, 1))])

    # calculate if the example that is to be initialized is even
    isEven = torch.tensor([float((nextData[0] + nextData[1] * 2 + nextData[2] * 4 + nextData[3] * 8) % 2 == 0)])

    # initialize [example, answer]
    initialData.append([nextData, isEven])


# the net
# has three layers
# takes in tensor of size 4, 4 bits representing an int
# outputs a tensor with a single float, one boolean representing even or odd
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


net = Net()  # make a net

criterion = nn.MSELoss()  # loss function, the most trivial one is used in this NN
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # optimizer, honestly I'm unsure

for data in initialData:  # iterate through data
    input, answer = data  # get current example and answer to it

    optimizer.zero_grad()  # ???

    output = net(input)  # run it through the net
    loss = criterion(output, answer)  # calculate error
    loss.backward()  # do stuff
    optimizer.step()  # do stuff

    print(output, answer)
