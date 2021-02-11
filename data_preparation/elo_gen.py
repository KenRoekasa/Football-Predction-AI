import csv
import sys

import numpy as np


import json
import pandas as pd

from elo import elo

if len(sys.argv) == 3:
    elo_ratings = {}

    league_csv = sys.argv[1]

    elorating_json = sys.argv[2]
    try:
        with open(elorating_json) as json_file:
            elo_ratings = json.load(json_file)
    except:
        pass


    matches = pd.read_csv(league_csv)


    for i, match in matches.iterrows():
        # if the team name isn't in the pi-rating dictionary add it in with an initial value
        if match.loc['home team'] not in elo_ratings:
            elo_ratings[match['home team']] = 1000


    print(elo_ratings)


    matches['date'] = pd.to_datetime(matches["date"])
    sorted_matches = matches.sort_values(by=['date']).copy()




    for i, row in sorted_matches.iterrows():
        teama = row['home team']
        teamb = row['away team']
        ascore = row['home score']
        bscore = row['away score']

        Ra, Rb = elo(int(elo_ratings[teama]), int(elo_ratings[teamb]), int(ascore), int(bscore))

        # Set in the elo in the dictionary
        elo_ratings[teama] = Ra
        elo_ratings[teamb] = Rb


        # print(row)
        # Add the elo to the table

        sorted_matches._set_value(i, 'home elo', Ra)
        sorted_matches._set_value(i, 'away elo', Rb)
        # row['home elo'] = Ra
        # print(row)
        # row['away elo'] = Rb



    # write back into the csv
    sorted_matches.to_csv(league_csv,index=False)


    with open(elorating_json, 'w') as fp:
        json.dump(elo_ratings, fp)

else:
    print("Invalid arguments elo_gen.py [csvfile] [elorating csv file name]")
