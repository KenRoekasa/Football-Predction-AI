import csv

filename = "../data/whoscored/column.csv"
with open(filename, 'r+', newline='') as myfile:
    csv_reader = csv.reader(myfile, delimiter=',')
    for row in csv_reader:
        columns = row
new_columns = []
for i, item in enumerate(columns):

    if i > 5:
        new_columns.append("home " + item.lower())
        new_columns.append("away " + item.lower())
    else:
        new_columns.append(item.lower())

with open(filename, 'w+', newline='') as myfile:
    wr = csv.writer(myfile, delimiter=',')
    wr.writerow(new_columns)


