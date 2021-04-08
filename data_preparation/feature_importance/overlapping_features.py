import numpy as np
import pandas as pd


def extract_top_features(top_number, csv):
    df = pd.read_csv(csv, names=['feature', 'importance'], header=None)
    df.sort_values(by='importance', ascending=False, inplace=True, ignore_index=True)
    df = df.iloc[:top_number]
    df = df['feature']
    top_list = df.tolist()
    return top_list



extract_number = 100

l1 = extract_top_features(extract_number,'decisiontree.csv')
# l2 = extract_top_features(extract_number,'kneighbour.csv')
l3 = extract_top_features(extract_number,'extratreeclassifer.csv')
l4 = extract_top_features(extract_number,'randomforest.csv')

# find overlap
print(list(set(l1) &set(l3)&set(l4)))


def features_to_numpy(csv):
    df = pd.read_csv(csv, names=['feature', 'importance'], header=None)

    df.sort_values(by='feature', ascending=False, inplace=True, ignore_index=True)

    columns = df['feature']

    df = df['importance']
    top_list = df.to_numpy()
    columns = columns.to_numpy()
    print(columns)
    return top_list, columns

n1,columns = features_to_numpy('decisiontree.csv')
n2,_ = features_to_numpy('extratreeclassifer.csv')
n3,_ = features_to_numpy('randomforest.csv')


final_np = np.array([n1,n2,n3])

sum_np = final_np.sum(axis=0)

combine_np = np.array([columns,sum_np])

table = pd.DataFrame(data=combine_np).T
table.sort_values(by=1, ascending=False, inplace=True, ignore_index=True)
table = table[0]

table.to_csv('sum_importance.csv',index=False)