import csv

def read_data_from_file(filepath):
    with open(filepath, "r") as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=",")

        cleanList = list()

        row_count = 0
        for row in csv_reader:
            cleanList.append([row_count, row["heartRate"]])
            row_count += 1

    return cleanList

def write_to_file(dataTuple_to_write, filepath):
    with open(filepath, "w+", newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        csv_writer.writerow(['Index', 'Heartrate'])

        for each in dataTuple_to_write:
            csv_writer.writerows([each])


data = read_data_from_file("test.csv")

write_to_file(data, "clean_file.csv")
