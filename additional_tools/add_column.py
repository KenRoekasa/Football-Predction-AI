import glob

import pandas as pd

import os
path = "D:/Desktop/kxr758/data/whoscored/ligue1/datascraped/"
season_label=['13/14','14/15','15/16','16/17','17/18','18/19','19/20']
owd = os.getcwd()
extension = 'csv'
os.chdir(path)
result = glob.glob('*.{}'.format(extension))
for i, r in enumerate(result):
        league = 'ligue1'
        season = season_label[i]
        df = pd.read_csv(r)
        df['league'] = league
        df['season'] = season
        df.to_csv(r, index=False)


