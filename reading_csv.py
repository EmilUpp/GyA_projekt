""" A File for reading csv files"""

import csv
from Decorators import timing


def calculate_time_in_bed(data):
    """
    Finds the interval the subject is sleeping by ignoring smaller gaps in pulse values
    Calculates datapoints between the two longest holes, a hole is a series of zero not interupted
    by more than 5 non zero numbers
    """

    # Save the indexes between two longest holes
    longest_hole = [0, 0]
    seconds_longest_hole = [0, 0]

    # Current hole start and current index
    current_hole_start = 0
    current_hole_end = 0

    # Loop the data
    for index, heartRate in data:
        # If datapoint is zero
        if heartRate == "0":
            continue

        # Otherwise
        elif (heartRate != "0" and non_zero_gap_size(data, index, 10) > 5):
            # Set endpoint to current point
            current_hole_end = index

            # Check if longer than longest
            if current_hole_end - current_hole_start > longest_hole[1] - longest_hole[0]:
                # Second longest to prior
                seconds_longest_hole = longest_hole

                # Set longest to this
                longest_hole = [current_hole_start, current_hole_end]

            # Check if longer than only the second
            elif current_hole_end - current_hole_start > seconds_longest_hole[1] - seconds_longest_hole[0]:
                seconds_longest_hole = [current_hole_start, current_hole_end]

            # Set current hole start to current index
            current_hole_start = index

    # Fixes the last hole
    # Set endpoint to current point
    current_hole_end = len(data)

    # Check if longer than longest
    if current_hole_end - current_hole_start > longest_hole[1] - longest_hole[0]:
        # Second longest to prior
        seconds_longest_hole = longest_hole

        # Set longest to this
        longest_hole = [current_hole_start, current_hole_end]

    # Check if longer than only the second
    elif current_hole_end - current_hole_start > seconds_longest_hole[1] - seconds_longest_hole[0]:
        seconds_longest_hole = [current_hole_start, current_hole_end]

    print("Longest: ", longest_hole)
    print("Second longest: ", seconds_longest_hole)

    # The longest comes before the second
    if longest_hole[1] < seconds_longest_hole[0]:
        return seconds_longest_hole[0] - longest_hole[1]

    # The second longest comes before the longest
    else:
        return longest_hole[0] - seconds_longest_hole[1]


def non_zero_gap_size(data, current_index, length):
    index = current_index
    gap_size = 0

    while index - current_index < length:
        if data[index][1] != "0":
            index += 1
            gap_size += 1
        else:
            break

    return gap_size


@timing
def read_data_from_file(filepath, data_row):
    """Reads specified datarow from file"""

    with open(filepath, "r") as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=",")

        clean_list = list()

        for row in csv_reader:
            heart_rate = row[data_row]

            if int(heart_rate) < 30:
                heart_rate = "0"

            clean_list.append([row["recordedAt"], heart_rate])

    return clean_list


def write_to_file(data_tuple_to_write, filepath):
    """Writes a tuple to a specified file"""

    with open(filepath, "w+", newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        csv_writer.writerow(['recordedAt', 'heartRate'])

        for each in data_tuple_to_write:
            csv_writer.writerows([each])


if __name__ == "__main__":
    data = read_data_from_file("test.csv", "heartRate")

    write_to_file(data, "clean_file.csv")
