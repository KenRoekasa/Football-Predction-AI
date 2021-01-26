import csv

from elo import elo
import json

elo_ratings = {}

try:
    with open('data/elorating.json') as json_file:
        elo_ratings = json.load(json_file)
except:
    pass

results = []

with open('data/england-premier-league-17-6.csv', mode='r') as csv_file:
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

with open("data/england-premier-league-17-6.csv", 'w+', newline='') as myfile:
    wr = csv.writer(myfile, delimiter=',')
    wr.writerows(results)

with open('data/elorating.json', 'w') as fp:
    json.dump(elo_ratings, fp)

# if teama in elo_ratings:
#     row.append(elo_ratings[teama])
# else:  # generate new elo rating
#     elo_ratings[teama] = 1000
#     row.append(1000)
#
# if teamb in elo_ratings:
#     row.append(elo_ratings[teamb])
# else:  # generate new elo rating
#     elo_ratings[teamb] = 1000
#     row.append(1000)
