import math
import random


def generate_one_night(length, end_value, msPerIndex):
    night_data = [0 for _ in range(length)]

    test_wake_up_time_index = math.floor(length * (random.random() * (0.9 - 0.7) + 0.7))

    increment = end_value / test_wake_up_time_index

    for i in range(test_wake_up_time_index):
        night_data[i] = round(increment * i)

    return [test_wake_up_time_index * msPerIndex, night_data]


def generate_data(batch_size, length, end_value, msPerIndex):
    data_set = []

    for _ in range(batch_size):
        data_set.append(generate_one_night(length, end_value, msPerIndex))

    return data_set


def sample_pop(list_to_sample: list, sample_amount):
    samples = random.sample(list_to_sample, sample_amount)

    for sample in samples:
        list_to_sample.remove(sample)

    return list_to_sample, samples


if __name__ == "__main__":
    test_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    print(sample_pop(test_list, 20))

    """
    for _ in range(10):
        test_data = generate_data(2000, 80, 20000)

        print(test_data[0])
        print(test_data[1])
        print()

    for each in generate_data(2000, 80, 200000)[1]:
        print(each)
    """
