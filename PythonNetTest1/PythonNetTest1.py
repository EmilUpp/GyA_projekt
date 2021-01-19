import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

import os.path as path
import math
import random

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

#Train the net with a tensor of data representing one chunked night
#Takes the net, optimiser and criterion
#Also takes data points (tensor) and sleep duration in milliseconds(int) as two seperate parameters
#Returns void but changes the net according to result from calculation and loss function
def train_net_one_night(net, criterion, optimizer, oneNightTensor, wakeUpTime):

    optimizer.zero_grad()

    guess = net(oneNightTensor)
    loss = criterion(guess, wakeUpTime)
    loss.backward()
    optimizer.step()

#Converts time (ms) to index
#Takes time in milliseconds (int)
#Returns an index (int)
#As every element in data tensor represents a interval of a constant amount of milliseconds,
#the operation is (time / ms per element)
#It is also floored as the index of a interval is equal to the lowest time in the interval / ms per element
def get_index_from_time(time):
    msPerElement = 20000
    return math.floor(time / msPerElement)

#Chunks a night
#Takes night duration in milliseconds (int) and data points (tensor) as two seperate variables
#Alse takes length of the wished chunk in milliseconds (int)
#Returns a new tensor which is a copy of data points but with every value after chunktime set to zero
#Will chunk at wakeuptime if it precedes chunktime
def chunk(wakeUpTime, nightData, chunkTime):
    #Copy tensor as to not change the original data
    #unsure if this is necessary, is there a way to make constants in python?
    chunkedData = nightData

    #Make sure we never chunk after wakeup time as this would include obstructuary data
    if chunkTime > wakeUpTime:
        chunkTime = wakeUpTime

    #Set all values between chunkindex and the end to zero
    i = get_index_from_time(chunkTime)
    while i < chunkData.size():
        chunkedData[i] = 0
        i += 1
    return chunkedData

#Trains the net with a set of nights.
#Takes the net, criterion and optimizer.
#
#Also takes a tensor where each element represents a night and contains 
#the correct answer or the duration of sleep and a tensor of all data values that night.
#
#Returns void.
#--
#The function will then split up the data in different "chunks", representing the net
#guessing att different times during the night.
#
#These will then be randomly run through the net and it will be changed according to 
#result and loss function.
def train_net(net, criterion, optimizer, allNightsTensor):
    #Temporary: declare how long the nights are and how long the chunks are
    chunkDifference = 3600000
    totalNightLength = 12 * 3600000

    #Declare list for storing all chunked up nights
    allTrainingChunks = []

    #For every night
    i = 0
    while i < allNightsTensor.size():
        #chunk until we are trying to chunk past the end
        j = chunkDifference
        while j <= totalNightLength:
            #Chunk and put it as a tensor with [wakeUpTime, [newlychunkedDataTensor]] into allTrainingChunks 
            allTrainingChunks.append(torch.tensor(
                allNightsTensor[i][0], chunk(allNightsTensor[i][0], allnightsTensor[i][1], j)))
            #Step to next chunking time, use iteration variable to keep track of this
            j += chunkDifference
        i += 1

    #Shuffle all chunks
    random.shuffle(allTrainingChunks)
    #run them all through the net
    i = 0
    while i < allTrainingChunks:
        train_net_one_night(net, criterion, optimizer, allTrainingChunks[i][1], allTrainingChunks[i][0])


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
