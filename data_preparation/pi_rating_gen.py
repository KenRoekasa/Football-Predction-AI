import csv
import sys

import json
import pandas as pd

from pi_rating import pi_rating


def pi_rating_gen(league_csv, pi_json):
    pi_ratings = {}
    try:
        with open(pi_json) as json_file:
            pi_ratings = json.load(json_file)
    except:
        pass
    matches = pd.read_csv(league_csv)
    for i, match in matches.iterrows():
        # if the team name isn't in the pi-rating dictionary add it in with an initial value
        if match.loc['home team'] not in pi_ratings:
            pi_ratings[match['home team']] = {}
            pi_ratings[match['home team']]['home'] = 0
            pi_ratings[match['home team']]['away'] = 0
    matches['date'] = pd.to_datetime(matches["date"], dayfirst=True)
    sorted_matches = matches.sort_values(by=['date']).copy()
    print(pi_ratings)
    for i, row in sorted_matches.iterrows():
        # apart from the header

        teama = row['home team']
        teamb = row['away team']
        ascore = row['home score']
        bscore = row['away score']
        # print(elo_ratings)

        Rah, Raa, Rbh, Rba = pi_rating(float(pi_ratings[teama]['home']), float(pi_ratings[teama]['away']),
                                       float(pi_ratings[teamb]['home']),
                                       float(pi_ratings[teamb]['away']), int(ascore), int(bscore))

        pi_ratings[teama]['home'] = Rah
        pi_ratings[teama]['away'] = Raa
        pi_ratings[teamb]['home'] = Rbh
        pi_ratings[teamb]['away'] = Rba

        sorted_matches._set_value(i, 'home home pi rating', Rah)
        sorted_matches._set_value(i, 'home away pi rating', Raa)
        sorted_matches._set_value(i, 'away home pi rating', Rbh)
        sorted_matches._set_value(i, 'away away pi rating', Rba)

        sorted_matches._set_value(i, 'home pi rating', float((Rah + Raa) / 2))
        sorted_matches._set_value(i, 'away pi rating', float((Rbh + Rba) / 2))
    # write back into the csv
    sorted_matches.to_csv(league_csv, index=False)
    with open(pi_json, 'w') as fp:
        json.dump(pi_ratings, fp)


if __name__ == '__main__':
    if len(sys.argv) == 3:
        pi_rating_gen(sys.argv[1], sys.argv[2])

    else:
        print("Invalid arguments pi_rating_gen.py [csvfile] [elorating csv file name]")
