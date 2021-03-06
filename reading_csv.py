"""
A file for reading csv files
"""

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
        elif heartRate != "0" and non_zero_gap_size(data, index, 10) > 5:
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

    # All values below are set to zero
    min_value_threshold = 30

    with open(filepath, "r") as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=",")

        clean_list = list()

        for row in csv_reader:
            heart_rate = row[data_row]

            if int(heart_rate) < min_value_threshold:
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


def write_2d_list_to_file(list_to_write, file_path):
    with open(file_path, "w") as file_handler:
        if type(list_to_write[0]) != list:

            file_handler.write("\n".join([str(x) for x in list_to_write]))
        else:
            for row in list_to_write:
                file_handler.write(",".join([str(x) for x in row]) + "\n")


def write_list_to_file(list_to_write, file_path):
    with open(file_path, "w") as file_handler:
        for element in list_to_write:
            file_handler.write(str(element)+"\n")


def append_list_to_file(list_to_append, file_path):
    with open(file_path, "r") as file_handler:
        new_list = list()
        read_lines = file_handler.readlines()

        for index, element in enumerate(list_to_append):
            try:
                new_list.append(str(read_lines[index].strip()) + "," + str(element) + "\n")
            except IndexError:
                new_list.append("," + str(element) + "\n")

    with open(file_path, "w") as file_handler:
        for line in new_list:
            file_handler.write(line)


if __name__ == "__main__":
    test_2_d = [[1, 2, 3, 4, 5, 6], [4, 5, 1, 123, 123], [123, 123, 123, 123]]
    test_list = [1, 2, 3, 4, 5]

    write_2d_list_to_file(test_list, "test_write.txt")