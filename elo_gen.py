import csv

from elo import elo
import json

elo_ratings = {}
league__csv = "data/germany-bundesliga-35-2.csv"
try:
    with open('data/elorating.json') as json_file:
        elo_ratings = json.load(json_file)
except:
    pass

results = []

with open(league__csv, mode='r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        results.append(row)

        if row[1] not in elo_ratings:
            elo_ratings[row[1]] = 1000

print(elo_ratings)

for row in results:
    teama = row[1]
    teamb = row[2]
    ascore = row[4]
    bscore = row[5]
    # print(elo_ratings)

    Ra, Rb = elo(int(elo_ratings[teama]), int(elo_ratings[teamb]), int(ascore), int(bscore))

    elo_ratings[teama] = Ra
    elo_ratings[teamb] = Rb

    row.append(Ra)
    row.append(Rb)

print(elo_ratings)

with open(league__csv, 'w+', newline='') as myfile:
    wr = csv.writer(myfile, delimiter=',')
    wr.writerows(results)

with open('data/elorating.json', 'w') as fp:
    json.dump(elo_ratings, fp)
