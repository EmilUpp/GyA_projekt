import math
import random


def generate_data(length, end_value):
    data_set = [0 for _ in range(length)]

    test_wake_up_time_index = math.floor(length * (random.random() * (0.9-0.7) + 0.7))

    increment = end_value / test_wake_up_time_index

    for i in range(test_wake_up_time_index):
        data_set[i] = increment * i * (1 + (0.05 - random.random() * 0.1))

    return data_set

if __name__ == "__main__":
    #test = [13, 15, 17]

    #test[-1] = str(test[-1]) + ","

    #print("".join(str(test)[1:-1]).replace("'", "").replace(" ", ""))

    test_data = generate_data(2000, 80)

    for value in test_data:
        print(value)