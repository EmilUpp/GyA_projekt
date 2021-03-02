import torch

from DataCleanup import full_data_formatting
from NightExtractor import read_excel_file, separate_nights
from PythonNetTestFolder import PythonNetTest1
from reading_csv import read_data_from_file, write_list_to_file, append_list_to_file
from TestDataGenerator import generate_data


def eval_net(net, night_to_eval):
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

        performance = round(int(chunk[0] - guess) / 1000 / 60)

        guesses.append(performance)

    net.train()

    return guesses



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

    net_guesses = eval_net(net, formatted_data_nights[18])

    write_list_to_file(net_guesses, "reference_night.txt")
    append_list_to_file(formatted_data_nights[18][1], "reference_night.txt")
