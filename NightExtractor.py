import csv
import datetime
from reading_csv import read_data_from_file


def format_date_time(date_string, time_string):
    """Combines the date and time string and convert them to a datetime object"""

    date = [int(x) for x in date_string.split("-")]
    time = [int(x) for x in time_string.split(".")]

    date_time = date + time

    formatted_date_time = datetime.datetime(date_time[2], date_time[1], date_time[0],
                                       date_time[3], date_time[4])

    return formatted_date_time


def format_time(time_string):
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

        # Gå igenom datan
        for row in csv_reader:

            # Ignore nights awoken by alarm
            if row["byAlarm"].lower() != "nej":
                continue

            # Reads data fields
            bed_time = row["bedTime"]
            wake_up_time = row["wakeUpTime"]
            wake_up_date = row["date"]

            # Converts wakeup time
            wake_up_time = format_date_time(wake_up_date, wake_up_time)
            wake_up_time = convert_to_epoch(wake_up_time)

            # Converts bedtime
            bed_time = format_date_time(wake_up_date, bed_time)

            # Checks if bedtime past midnight
            if bed_time.hour > 12:
                bed_time -= datetime.timedelta(days=1)

            bed_time = convert_to_epoch(bed_time)

            # Convert sleep duration
            sleep_duration = wake_up_time-bed_time

            # Skapa en lista av tuples med bedtime och Wakeup time
            read_data_list.append((
                bed_time,
                wake_up_time,
                sleep_duration
            ))

    return read_data_list


def seperate_nights(excel_data, sleep_data):
    # Lista med alla nätters data
    separated_nights_data = []

    # Lista med nuvarande nattens puls
    current_night_data = []

    # Current night bedtime
    current_night_bedtime = excel_data[0][0]

    # Nuvarande natt wakeup
    current_night_wakeup = excel_data[0][1]

    # Gå igenom Sömndatan
    for recorded_at, pulse in sleep_data:
        # Convert to int
        recorded_at, pulse = int(recorded_at), int(pulse)

        # Om passerar nuvarande wakeup och det finns data i nuvarande natt
        if (recorded_at > current_night_wakeup and len(current_night_data) > 0):
            # Lägg till natten till alla nätter
            separated_nights_data.append(current_night_data)

            # Nollställ nuvarande natt
            current_night_data = []

        # If passes a night bedtime and waiting to start a new
        if (recorded_at > current_night_bedtime and len(current_night_data) == 0):
            current_night_bedtime = excel_data[len(separated_nights_data)][0]
            current_night_wakeup = excel_data[len(separated_nights_data)][1]

        # Om tiden är efter nuvarande bedtime men innan nuvarande natt wakeup
        if (recorded_at > current_night_bedtime and recorded_at < current_night_wakeup):
            # Lägg till pulsen
            current_night_data.append(pulse)

    return separated_nights_data

if __name__ == "__main__":
    # excel_data
    excel_data = read_excel_file("Vår sömn - Abbes.csv")

    # Sleep data
    sleep_data = read_data_from_file("CompleteDataSet.csv", "heartRate")

    separated_nights = seperate_nights(excel_data, sleep_data)

    for night in separated_nights:
        print(night)
        print()

    print(len(separated_nights))
    print(len(excel_data))
