import csv

with open("test.csv", "r") as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=",")

    cleanList = list()

    row_count = 0
    for row in csv_reader:
        if (row["heartRate"] != 0):
            cleanList.append((row_count, row["heartRate"]))

        row_count += 1

print(cleanList)
