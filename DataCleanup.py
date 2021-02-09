"""
Functions for cleaning up and simplifying the data for processing
"""

from Decorators import timing


def calculate_mean(data_chunk, start_time, time_interval, debug=False):
    """
    Calculates the mean of a certain interval of datapoints

    :param data_chunk: 2d list, [(recorded_at1, pulse1), (recorded_at2, pulse2)]
    :param start_time: int, time the chunks starts in epoch time milliseconds
    :param time_interval: int, interval of each chunk, milliseconds
    :param debug: bool, prints all chunks and the calculated mean
    :return: int, the mean pulse value for the interval
    """

    mean_value = 0

    # check if there was any point in the actual chunk
    if not len(data_chunk) > 1:
        # return the first value after the chunk i.e the only one in the chunk
        return data_chunk[0][1]

    # last point is instantiated as recorded at start
    last_point = (start_time, None)
    zero_counter = 0

    # loop through points except last one
    for recorded_at, pulse in data_chunk[:-1]:
        if zero_counter > len(data_chunk) / 1.5:
            return 0

        # hmmmm might want to change this later
        if pulse == 0:
            zero_counter += 1
            continue

        # calculate percentage of chunk for point
        percentage_of_chunk = (recorded_at - last_point[0]) / time_interval

        # add pulse * percentage of chunk
        mean_value += pulse * percentage_of_chunk

        # Set new last point
        last_point = (recorded_at, pulse)

    # Last one is calculated to the edge of the chunk
    if data_chunk[-1][1] != 0:
        percentage_of_chunk = ((start_time + time_interval) - last_point[0]) / time_interval
        mean_value += data_chunk[-1][1] * percentage_of_chunk

        if mean_value < 0:
            print(round(mean_value), round(percentage_of_chunk))
            print(data_chunk)
            print(start_time + time_interval, last_point[0])
            print()

    else:
        # If the first one beyond the interval is zero the last non zero is extended
        percentage_of_chunk = ((start_time + time_interval) - last_point[0]) / time_interval
        try:
            mean_value += last_point[1] * percentage_of_chunk
        except TypeError:
            pass

    mean_value = round(mean_value)

    if debug:
        print()
        print("mean: " + str(mean_value), end="\n")
        for recorded_at, pulse in data_chunk:
            print(round((recorded_at - start_time) / 1000, 1), pulse, end=", ")
        print()

    return mean_value


# TODO fix potential problem with succesive empty chunks being counted as one empty chunk

# Calculate rolling mean
def calculate_rolling_mean(pulse_data, time_interval, debug=False):
    """ Calculates the rolling mean of the input values and creates a new list with
    the mean values uniformly spaced over time, how much time is between the values
    is set by the time interval parameter

    :param pulse_data: list of pulses, (recordedAt, pulse)
    :param time_interval: int, interval of each chunk, milliseconds
    :param debug: bool, prints all values
    :return: a list of of pulse values uniformly spaced over time
    """

    calculated_mean_list = []

    # start time = time of first recording
    start_time = pulse_data[0][0]
    # interval end time = start_time + time_interval
    end_time = start_time + time_interval

    # current_chunk_list
    current_chunk = []

    # loop every datapoint
    for index, (recorded_at, pulse) in enumerate(pulse_data):

        # if point is after interval end time
        if recorded_at > end_time:
            if divmod(recorded_at - end_time, time_interval)[0] > 0 and debug:
                print(divmod(recorded_at - end_time, time_interval)[0])
                print((recorded_at - pulse_data[0][0]) / 1000)
                print(len(current_chunk))
                print()

            # calculate
            mean_value = calculate_mean(current_chunk, start_time, time_interval, debug)

            # add mean to list
            calculated_mean_list.append(mean_value)

            # reset values
            # Since the last is after end it's added to the next one
            current_chunk = [(recorded_at, pulse)]
            start_time = end_time
            end_time = start_time + time_interval
        else:
            # Point is between start and end and is added
            current_chunk.append((recorded_at, pulse))

    return calculated_mean_list


def compare_accuracy(raw_data_set, different_intervals):
    """
    Calculates the rolling mean values for diffenr intervals and saves all in an csv for excel visulazations

    :param raw_data_set: (sleep_duration1, [(recorded_at1, pulse1), (recorded_at2, pulse2)]), One nights complete data
    :param different_intervals: List[int], A list of time intervals in milliseconds
    :return: None
    """

    # Calculate all accuracies and add to list
    multiple_level_data = list()
    multiple_level_data.append((1, raw_data_set))

    for interval in different_intervals:
        multiple_level_data.append((interval, calculate_rolling_mean(raw_data_set, interval)))

    # Open file
    with open("accuracy_comparison.csv", "w+") as file_handler:
        # Write first line
        file_handler.write(",".join(str(x[0] / 1000) + " sec" + "," for x in multiple_level_data) + "\n")

        # Enumerate indexes in raw
        for index in range(len(raw_data_set)):
            file_handler.write(str((multiple_level_data[0][1][index][0] - multiple_level_data[0][1][0][0]) / 1000)
                               + "," + str(multiple_level_data[0][1][index][1]))

            # Try and write corresponding index from other interval
            for interval, data in multiple_level_data[1:]:
                try:
                    file_handler.write("," + str(index * int(interval) / 1000) + "," + str(data[index]))
                except IndexError:
                    file_handler.write(",#SAKNAS,#SAKNAS")
                    pass

            file_handler.write("\n")

            # Try each and write to file


# Add trailing zero to specified length
def add_trailing_zeros(un_trailed_data, target_length):
    """Adds trailing zeroes to make the list the target length"""

    # Cut the list of it's longer then target length
    if len(un_trailed_data) > target_length:
        return un_trailed_data[:target_length]

    un_trailed_data.extend([0 for _ in range(target_length - len(un_trailed_data))])
    return un_trailed_data


def full_data_formatting(one_night_data_set, time_interval, target_length):
    """
    Formats the data by reducing the number of points by calculating the
    rolling average of the time interval specified. If the data is shorter
    then specified length trailing zeros are added

    :param one_night_data_set: [sleep_duration, [pulse_data]], dataset for one night
    :param time_interval: int, time interval to calculate rolling mean
    :param target_length: int, length of array of data points
    :return: [sleep_duration, int[pulse_data]]
    """
    sleep_length, pulse_data = one_night_data_set

    mean_data = calculate_rolling_mean(pulse_data, time_interval)

    mean_trailed_data = add_trailing_zeros(mean_data, target_length)

    return [sleep_length, mean_trailed_data]


if __name__ == "__main__":
    compare_accuracy([(0, 10), (1, 10), (2, 7)], [20, 60])
