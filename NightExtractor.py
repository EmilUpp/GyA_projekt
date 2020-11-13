"""
A file for extracting and separating different nights data from one large csv file
"""

import csv
import datetime
from reading_csv import read_data_from_file
from Decorators import timing


def format_date_time(date_string, time_string):
    """Combines the date and time string and convert them to a datetime object"""

    date = [int(x) for x in date_string.split("-")]
    time = [int(x) for x in time_string.split(".")]

    date_time = date + time

    formatted_date_time = datetime.datetime(date_time[2], date_time[1], date_time[0],
                                            date_time[3], date_time[4])

    return formatted_date_time


def format_time(time_string):
    """ Converts the time to a datetime object"""

    time = [int(x) for x in time_string.split(".")]

    formatted_time = datetime.time(time[0], time[1])
    return formatted_time


def convert_to_epoch(formatted_time):
    """Convert to epoch time, formatted_time is a datetime object"""

    return formatted_time.timestamp() * 1000


def read_excel_file(excel_file_path):
    """
    Reads the data from an excel sheet for the parameters bedTime and WakeUpTime and converts them to epoch time
    :param excel_file_path: string, filepath to the file to read
    :return: a tuple, format tuple(bedTime, wakeUpTime, sleepDuration) dates in epoch time (millisecond) and duration in milliseconds
    """

    # Läs in nätterna från excel
    with open(excel_file_path, "r") as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=",")

        read_data_list = list()

        for row in csv_reader:

            # Ignore nights awoken by alarm
            if row["byAlarm"].lower() != "nej":
                continue

            # Read data fields
            bed_time = row["bedTime"]
            wake_up_time = row["wakeUpTime"]
            wake_up_date = row["date"]

            # Converts wakeup time
            wake_up_time = format_date_time(wake_up_date, wake_up_time)
            wake_up_time = convert_to_epoch(wake_up_time)

            # Converts bedtime
            bed_time = format_date_time(wake_up_date, bed_time)

            # Checks if bedtime past midnight if so subtract 24 hours
            if bed_time.hour > 12:
                bed_time -= datetime.timedelta(days=1)

            bed_time = convert_to_epoch(bed_time)

            # Convert sleep duration
            sleep_duration = wake_up_time - bed_time

            # Skapa en lista av tuples med bedtime och Wakeup time
            read_data_list.append((
                bed_time,
                wake_up_time,
                sleep_duration
            ))

    return read_data_list


# TODO fix edge case with wintertime change
def separate_nights(excel_data, sleep_data, debug_mode=False):
    """
    Seperates nights according to inout data from excel sheet
    :param excel_data: csv file
    :param sleep_data: list of pulses, (recordedAt, pulse)
    :param debug_mode: prints error messages
    :return: a list containing tuples of all nights with pulses and sleep duration
            ((pulse_data1, sleep_duration1),(pulse_data2, sleep_duration2)
    """
    separated_nights_data = []

    # Counts nights read from excel
    # This needs to be separate because of wintertime bug
    night_counter = 0

    # List of current night pulses
    current_night_data = []

    # Current night bedtime instantiated to first night
    current_night_bedtime = excel_data[0][0]

    # Current night wakeup instantiated to first night
    current_night_wakeup = excel_data[0][1]

    # Loop through data
    for recorded_at, pulse in sleep_data:
        # Convert to int
        recorded_at, pulse = int(recorded_at), int(pulse)

        # If passed current night wakeup
        if (recorded_at > current_night_wakeup):

            # Add night to list
            # Checks if there any data in the list, due to wintertime bug 25 october have 0 pulses
            if (len(current_night_data) > 0):
                separated_nights_data.append([excel_data[night_counter][2], current_night_data])

            # Reset
            current_night_data = []

            night_counter += 1

            try:
                current_night_bedtime = excel_data[night_counter][0]
                current_night_wakeup = excel_data[night_counter][1]
            except IndexError:
                if debug_mode:
                    print("Index error: "
                          + str(current_night_bedtime) + " "
                          + str(current_night_wakeup) + " "
                          + str(len(current_night_data))
                          )
                break

        # If time is between bedtime and wakeup
        if (recorded_at > current_night_bedtime and recorded_at < current_night_wakeup):
            current_night_data.append(pulse)

    return separated_nights_data


def align_data(separated_data):
    """ Aligns all data and writes it to a csv file for comparisons """

    with open("combined_data.csv", "w+") as file_handler:
        file_handler.write("".join(str([x for x in range(0, len(separated_data))])[1:-1]) + ",\n")

        combined = dict()

        for _, night in separated_data:
            for index, each in enumerate(night):
                combined.setdefault(str(index), [])
                combined[str(index)].append(each)

        for index, each in combined.items():
            file_handler.write("".join(str(each)[1:-1]) + ",\n")


if __name__ == "__main__":
    # Manual data with bedtime and wakeup
    excel_data = read_excel_file("Vår sömn - Abbes.csv")

    # Sensor data
    sleep_data = read_data_from_file("CompleteDataSet.csv", "heartRate")

    separated_nights = separate_nights(excel_data, sleep_data)
