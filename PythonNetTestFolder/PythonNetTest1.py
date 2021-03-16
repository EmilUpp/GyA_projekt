import math
import os.path as path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from DataCleanup import full_data_formatting
from Decorators import timing
from NightExtractor import read_excel_file, separate_nights
from PythonNetTestFolder import NetEvaluation
from TestDataGenerator import generate_data
from reading_csv import read_data_from_file, write_list_to_file, append_list_to_file, write_2d_list_to_file

# Saves the performance into groups depending on time left
performanceComparison = dict()
one_night_performance = list()
loss_list = []
loss_list_mean = []
eval_samples = list()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2000, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 25)
        # self.fc4 = nn.Linear(800, 50)
        self.fc4 = nn.Linear(25, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        x = self.fc4(x)
        return x

      
# Train the net with a tensor of data representing one chunked night
# Takes the net, optimiser and criterion
# Also takes data points (tensor) and sleep duration in milliseconds(int) as two separate parameters
# Returns void but changes the net according to result from calculation and loss function
def train_net_one_night(net, criterion, optimizer, oneNightTensor, wakeUpTime):
    optimizer.zero_grad()

    guess = net(oneNightTensor)
    loss = criterion(guess, wakeUpTime)
    loss.backward()
    optimizer.step()

    loss_list.append(loss.item() / math.pow(1000 * 60, 2))

    if len(loss_list) % 100 == 0:
        loss_list_mean.append(round(sum(loss_list[-100:])/100))

    group_performances(guess, wakeUpTime)

    # When only one for testing
    # performance = round(abs(int(wakeUpTime - guess) / 1000 / 60), 4)
    # one_night_performance.append([performance, round(abs(int(wakeUpTime) / 1000 / 60), 4)])


# Converts time (ms) to index
# Takes time in milliseconds (int)
# Returns an index (int)
# As every element in data tensor represents a interval of a constant amount of milliseconds,
# the operation is (time / ms per element)
# It is also floored as the index of a interval is equal to the lowest time in the interval / ms per element
def get_index_from_time(time, msPerElement):
    return math.floor(time / msPerElement)


def group_performances(guess, wakeUpTime):
    """Groups performances after how long before wakeup time it was made"""

    hour_in_ms = 3600000

    # Performance counted in minutes
    performance = round(abs(int(wakeUpTime - guess) / 1000 / 60), 2)

    # Different time interval
    if wakeUpTime < hour_in_ms:
        performanceComparison.setdefault("<1h", [])
        performanceComparison["<1h"].append(performance)
    elif wakeUpTime < hour_in_ms * 3:
        performanceComparison.setdefault("1-3h", [])
        performanceComparison["1-3h"].append(performance)
    elif wakeUpTime < hour_in_ms * 6:
        performanceComparison.setdefault("3-6h", [])
        performanceComparison["3-6h"].append(performance)
    else:
        performanceComparison.setdefault(">6h", [])
        performanceComparison[">6h"].append(performance)


def save_performance(performances):
    """Saves the performances to a csv file"""
    with open("performanceComparison.csv", "w+") as file_handler:
        # Writes the columnnames
        # file_handler.write(",".join(performances.keys()) + "\n")

        file_handler.write("<1h,1-3h,3-6h,>6h\n")

        combined = list()

        # Combines all data by building a list of all items with index n and appending list to main list
        for index, value in enumerate(performances["<1h"]):
            combined.append([value])

            # Tries to add latter and but they are shorter
            try:
                combined[index].append(performances["1-3h"][index])
            except IndexError:
                combined[index][-1] = str(combined[index][-1]) + ","

            try:
                combined[index].append(performances["3-6h"][index])
            except IndexError:
                combined[index][-1] = str(combined[index][-1]) + ","

            try:
                combined[index].append(performances[">6h"][index])
            except IndexError:
                combined[index][-1] = str(combined[index][-1]) + ","
            except KeyError:
                pass

        for each in combined:
            file_handler.write("".join(str(each)[1:-1]).replace("'", "").replace(" ", "") + "\n")


# Chunks a night
# Takes night duration in milliseconds (int) and data points (tensor) as two seperate variables
# Alse takes length of the wished chunk in milliseconds (int)
# Returns a new tensor which is a copy of data points but with every value after chunktime set to zero
# Will chunk at wakeuptime if it precedes chunktime
def chunk(wakeUpTime, nightData, chunkTime):
    # Copy tensor as to not change the original data
    # unsure if this is necessary, is there a way to make constants in python?
    chunkedData = nightData.copy()

    # Make sure we never chunk after wakeup time as this would include obstructuary data
    if chunkTime > wakeUpTime:
        chunkTime = wakeUpTime

    # Set all values between chunkindex and the end to zero
    i = get_index_from_time(chunkTime, 20000)

    # Marker?
    # chunkedData[i-1] = -1
    while i < len(chunkedData):
        chunkedData[i] = -1
        i += 1

    #print(chunkedData)
    return chunkedData


# Trains the net with a set of nights.
# Takes the net, criterion and optimizer.
#
# Also takes a tensor where each element represents a night and contains
# the correct answer or the duration of sleep and a tensor of all data values that night.
#
# Returns void.
# --
# The function will then split up the data in different "chunks", representing the net
# guessing att different times during the night.
#
# These will then be randomly run through the net and it will be changed according to
# result and loss function.
@timing
def train_net(net, criterion, optimizer, allNightsTensor, epochs, check_points=False):
    # Temporary: declare how long the nights are and how long the chunks are
    chunkDifference = 1000 * 60 * 5
    firstChunkOffset = 3600000 * 2
    totalNightLength = 10 * 3600000

    # Declare list for storing all chunked up nights
    allTrainingChunks = []

    # For every night
    i = 0
    while i < len(allNightsTensor):
        # Offset is no maximize performance in the end and not in the beginning
        j = chunkDifference + firstChunkOffset

        # chunk until wakeup and then twice again
        while j <= allNightsTensor[i][0] + chunkDifference * 2:
            # Calculate time from end of chunk to wakeup, never negative but 0 instead
            timeUntilWakeUp = max(allNightsTensor[i][0] - j, 0)

            # Chunk and put it as a tensor with [wakeUpTime, [newlychunkedDataTensor]] into allTrainingChunks
            allTrainingChunks.append((timeUntilWakeUp,
                                      chunk(allNightsTensor[i][0], allNightsTensor[i][1], j)))
            # Step to next chunking time, use iteration variable to keep track of this
            j += chunkDifference

        i += 1

    sample_size = round(len(allTrainingChunks) * 0.15)
    sample_chunks, allTrainingChunks = NetEvaluation.sample_pop(allTrainingChunks, sample_size)
    eval_samples.extend(sample_chunks)
    print(len(allTrainingChunks), len(sample_chunks), sample_size)

    for index, epoch in enumerate(range(epochs)):
        # Shuffle all chunks
        random.shuffle(allTrainingChunks)
        # run them all through the net
        i = 0
        while i < len(allTrainingChunks):
            train_net_one_night(net, criterion, optimizer,
                                torch.Tensor(allTrainingChunks[i][1]),
                                torch.Tensor([allTrainingChunks[i][0]]))
            i += 1

            if i % (min(round(len(allTrainingChunks)/3), 1000)) == 0:
                print("%d out of %d iterations completed" % (i, len(allTrainingChunks)))

        print("Epoch %d out of %d completed" % (index + 1, epochs))
        print("Epoch mean loss: " + str(round(sum(loss_list[round(-len(allTrainingChunks)):])
                                              /round(len(allTrainingChunks)))))
        print()

    print("Training complete")
    print("Total dataset: " + str(len(allTrainingChunks)))
    print("Total iterations: " + str(len(allTrainingChunks) * epochs))

        
if __name__ == "__main__":
    pathName = "testNet.pth"
    use_loaded_net = False

    # create a new net
    net = Net()

    # if there is a saved net
    if path.isfile(pathName) and use_loaded_net:
        # get saved net
        net.load_state_dict(torch.load(pathName))

        # Use only when not training
        # net.eval()

    # define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), 0.001, betas=(0.9, 0.999))

    # get data

    # Manual data with bedtime and wakeup
    excel_data = read_excel_file("Vår sömn - Abbes2.csv")

    # Sensor data
    sleep_data = read_data_from_file("PulseData11OctTo3Dec.csv", "heartRate")
    separated_nights = separate_nights(excel_data, sleep_data)

    # Format the data
    formatted_data_nights = list()
    for night in separated_nights:
        formatted_data_nights.append(full_data_formatting(night, 20000, 2000))

    test_data_set = generate_data(30, 2000, 80, 20000)

    # formatted_data_nights = formatted_data_nights[:4]

    # reference_night_index = random.randint(0, len(formatted_data_nights) - 1)
    # reference_night_index = 0

    # reference_night = formatted_data_nights.pop(reference_night_index)

    # train the net
    train_net(net, criterion, optimizer, formatted_data_nights, 10)

    # save the net
    torch.save(net.state_dict(), pathName)

    save_performance(performanceComparison)

    # net_guesses = NetEvaluation.eval_net_one_night(net, reference_night)
    net_sample_guesses = NetEvaluation.eval_net_on_samples(net, eval_samples)

    write_2d_list_to_file(net_sample_guesses, "performance_against_time.txt")

    """
    print()
    print("Reference night: " + str(reference_night_index))
    print("Night length: " + str(reference_night[0] / (1000 * 60 * 60)))
    print()

    write_list_to_file(net_guesses, "reference_night.txt")
    append_list_to_file(reference_night[1], "reference_night.txt")
    """
