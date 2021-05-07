import pandas as pd

backup_path = "D:/Desktop/kxr758/data/whoscored/backup/premierleague-20192020.csv"
path = "D:/Desktop/kxr758/data/whoscored/premierleague/datascraped/premierleague-20192020.csv"

df_bk = pd.read_csv(backup_path)
df = pd.read_csv(path)


df['date'] = df_bk['date']


df.to_csv(path,index=False)
