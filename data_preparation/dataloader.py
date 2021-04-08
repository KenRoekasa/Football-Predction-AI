import numpy
import pandas as pd
from sklearn.preprocessing import normalize, MinMaxScaler

# pd.set_option("display.max_rows", None, "display.max_columns", None)


def normalise_input_array(array, norm):
    x = numpy.copy(array)
    if norm == 'ratio':
        mean_array_sum = array[:, 1:-4:2] + array[:, 0:-4:2]
        with numpy.errstate(divide='ignore', invalid='ignore'):
            x[:, 0:-4:2] = numpy.true_divide(x[:, 0:-4:2], mean_array_sum)
            x[:, 1:-4:2] = numpy.true_divide(x[:, 1:-4:2], mean_array_sum)

            x[:, 0:-4:2] = numpy.nan_to_num(x[:, 0:-4:2], nan=0.5)
            x[:, 1:-4:2] = numpy.nan_to_num(x[:, 1:-4:2], nan=0.5)

    if norm == 'l1' or norm == 'l2':
        x = normalize(x, axis=1, norm=norm)
    elif norm == 'max':
        scaler = MinMaxScaler()
        scaler.fit(x[:, 0:-4])
        x[:, 0:-4] = scaler.transform(x[:, 0:-4])

    return x



# load a pickle file of the training data
def load_training_data(path, features, league, return_columns=False):
    training_data = pd.read_csv(path)
    if league != 'all':
        training_data = training_data[training_data['league'] == league]

    training_data.drop(columns=['league'], inplace=True)
    training_data['Outcome'] = pd.Categorical(training_data['Outcome'])
    training_data = training_data.astype({"Outcome": int})

    target = training_data.pop('Outcome')
    training_data.astype('float64').dtypes

    temp = []

    if features != []:  # everything is included

        for f in features:
            temp.append('home %s' % f)
            temp.append('away %s' % f)


    else:  # interleave features
        l1 = training_data.columns[:len(training_data.columns) // 2]
        l2 = training_data.columns[len(training_data.columns) // 2:]

        temp = [val for pair in zip(l1, l2) for val in pair]

    training_data = training_data[temp]

    #
    # home_pi_rating_column = training_data.pop('home pi rating')
    # away_pi_rating_column = training_data.pop('away pi rating')
    #
    # home_elo_column = training_data.pop('home elo')
    # away_elo_column = training_data.pop('away elo')

    x = training_data.to_numpy()

    y = target.to_numpy()

    if return_columns:
        return x, y, temp

    return x, y




    # merge_seasons('../data/whoscored/premierleague/datascraped','../all-premierleague.csv')
    # merge_seasons('../data/whoscored/laliga/datascraped', '../all-laliga.csv')
    # merge_seasons('../data/whoscored/seriea/datascraped', '../all-seriea.csv')
    # merge_seasons('../data/whoscored/bundesliga/datascraped', '../all-bundesliga.csv')
    # merge_seasons('../data/whoscored/ligue1/datascraped', '../all-ligue1.csv')
    #
    # merge_leagues()
