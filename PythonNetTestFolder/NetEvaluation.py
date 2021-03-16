import random

import torch

from DataCleanup import full_data_formatting
from NightExtractor import read_excel_file, separate_nights
from PythonNetTestFolder import PythonNetTest1
from reading_csv import read_data_from_file, write_list_to_file, append_list_to_file
from TestDataGenerator import generate_data


def eval_net_one_night(net, night_to_eval):
    """

    :param net: Net, the neural net model used
    :param night_to_eval: [sleep_duration, [pulses]], data for one night
    :return: list(), list of guesses
    """
    # Set to eval mode
    net.eval()

    chunkDifference = 1000 * 60 * 5
    firstChunkOffset = 3600000 * 2
    totalNightLength = 10 * 3600000

    chunks_to_evaluate = []
    guesses = []

    # chunk until we are trying to chunk past the end
    j = chunkDifference + firstChunkOffset
    while j <= night_to_eval[0] + chunkDifference * 2:
        # Calculate time from end of chunk to wakeup
        timeUntilWakeUp = max(night_to_eval[0] - j, 0)
        # Chunk and put it as a tensor with [wakeUpTime, [newlychunkedDataTensor]] into chunks_to_evaluate
        chunks_to_evaluate.append((timeUntilWakeUp,
                                   PythonNetTest1.chunk(night_to_eval[0], night_to_eval[1], j)))
        # Step to next chunking time, use iteration variable to keep track of this

        j += chunkDifference

    for chunk in chunks_to_evaluate:
        guess = net(torch.Tensor(chunk[1]))
        # print(guess.item())
        print(chunk[0])

        performance = round(int(chunk[0] - guess) / 1000 / 60)

        guesses.append(performance)

    net.train()

    return guesses


def eval_net_on_samples(net, list_of_samples):
    """
    Evaluates the net on random selected chunks and pairs them by sleep duration left

    :param net: Net, the neural net model used
    :param list_of_samples: list(), list of chunks to check
    :return: [[performance1, timeleft1], [performance2, timeleft2]]
    """

    net.eval()

    sample_guesses = list()

    for sample in list_of_samples:
        guess = net(torch.Tensor(sample[1]))

        performance = abs(round(int(sample[0] - guess) / 1000 / 60))

        sample_guesses.append([sample[0], performance])

    net.train()

    return sample_guesses


def sample_pop(list_to_sample: list, sample_amount):
    """
    Randomly selects and pops elements from a list

    :param list_to_sample: list, list to pick samples from
    :param sample_amount: int, how many samples
    :return:
    """

    try:
        samples = random.sample(list_to_sample, sample_amount)
    except ValueError:
        # If one tries and illegal sample size (to big or negative) the samples list is empty
        return list_to_sample, list()

    for sample in samples:
        list_to_sample.remove(sample)

    return samples, list_to_sample


if __name__ == "__main__":
    night_index_to_eval = 20

    # Manual data with bedtime and wakeup
    excel_data = read_excel_file("Vår sömn - Abbes2.csv")

    # Sensor data
    sleep_data = read_data_from_file("PulseData11OctTo3Dec.csv", "heartRate")
    separated_nights = separate_nights(excel_data, sleep_data)

    # Format the data
    formatted_data_nights = list()
    for night in separated_nights:
        formatted_data_nights.append(full_data_formatting(night, 20000, 2000))

    net = PythonNetTest1.Net()
    net.load_state_dict(torch.load("testNet.pth"))

    # net_guesses = eval_net(net, formatted_data_nights[night_index_to_eval])
    test_data = generate_data(5, 2000, 80, 20000)[0]

    net_guesses = eval_net_one_night(net, formatted_data_nights[0])

    write_list_to_file(net_guesses, "reference_night.txt")
    append_list_to_file(formatted_data_nights[0][1], "reference_night.txt")
