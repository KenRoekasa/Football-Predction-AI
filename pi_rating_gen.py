import csv

from pi_rating import pi_rating
import json

pi_ratings = {}
league__csv = "data/england-premier-league-17-2.csv"
ratings_json = 'data/pi_ratings.json'
try:
    with open(ratings_json) as json_file:
        pi_ratings = json.load(json_file)
except:
    pass

results = []

with open(league__csv, mode='r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for i, row in enumerate(csv_reader):
        results.append(row)

        # apart from the header
        if i > 0:
            # if the team name isn't in the pi-rating dictionary add it in with an initial value
            if row[1] not in pi_ratings:
                pi_ratings[row[1]] = {}
                pi_ratings[row[1]]['home'] = 0
                pi_ratings[row[1]]['away'] = 0

print(pi_ratings)

for i, row in enumerate(results):
    # apart from the header
    if i > 0:
        teama = row[1]
        teamb = row[2]
        ascore = row[4]
        bscore = row[5]
        # print(elo_ratings)

        Rah, Raa, Rbh, Rba = pi_rating(float(pi_ratings[teama]['home']), float(pi_ratings[teama]['away']),
                                       float(pi_ratings[teamb]['home']),
                                       float(pi_ratings[teamb]['away']), int(ascore), int(bscore))

        pi_ratings[teama]['home'] = Rah
        pi_ratings[teama]['away'] = Raa
        pi_ratings[teamb]['home'] = Rbh
        pi_ratings[teamb]['away'] = Rba

        row.append(Rah)
        row.append(Raa)
        row.append(Rbh)
        row.append(Rba)

print(pi_ratings)

with open(league__csv, 'w+', newline='') as myfile:
    wr = csv.writer(myfile, delimiter=',')
    wr.writerows(results)

with open(ratings_json, 'w') as fp:
    json.dump(pi_ratings, fp)
