import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

import os.path as path

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4000, 500)
        self.fc2 = nn.Linear(500, 50)
        self.fc3 = nn.Linear(50, 25)
        self.fc4 = nn.Linear(25, 8)
        self.fc5 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

def train_net_one_night(net, criterion, optimizer, oneNightTensor, wakeUpTime):

    optimizer.zero_grad()

    guess = net(oneNightTensor)
    loss = criterion(guess, wakeUpTime)
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    pathName = "testNet.pth"

    #create a new net
    net = Net()

    #if there is a saved net
    if path.isfile(pathName):
        #get saved net
        net.load_state_dict(torch.load(pathName))
        net.eval()

    #define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), 0.001, betas=(0.9, 0.999))

    #get data

    #train the net

    #save the net
    torch.save(net.state_dict(), pathName)
