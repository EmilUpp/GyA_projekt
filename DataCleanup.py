"""
Functions for cleaning up and simplifying the data for processing
"""


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
    else:
        # If the first on beyond the interval is zero the last non zero is extended
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
            print(round((recorded_at-start_time)/1000, 1), pulse, end=", ")
        print()

    return mean_value


# Calculate rolling mean
def calculate_rolling_mean(pulse_data, time_interval, debug=False):
    """
    Calculates the rolling mean of the input values and creates a new list with
    the mean values uniformly spaced over time, how much time is between the values
    is set by the time interval parameter

    :param pulse_data: list of pulses, (recordedAt, pulse)
    :param time_interval: int, interval of each chunk, milliseconds
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
        # add point to current chunk
        current_chunk.append((recorded_at, pulse))

        # if point is after interval end time
        if recorded_at > end_time:
            # calculate_mean()
            mean_value = calculate_mean(current_chunk, start_time, time_interval, debug)

            # add mean to list
            calculated_mean_list.append(mean_value)

            # reset values
            # The last one is after the ending and is therefore the first one in the next chunk
            current_chunk = []
            start_time = end_time
            end_time = start_time + time_interval

    return calculated_mean_list


# Add trailing zero to specified length
def add_trailing_zeros(un_trailed_data, target_length):
    pass