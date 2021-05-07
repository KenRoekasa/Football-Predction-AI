import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys


if __name__ == '__main__':

    data = pd.read_csv(sys.argv[1],header=None)

    sort = data.sort_values(data.columns[0])

    numpy_data = np.split(sort,2)

    numpy_data = np.hstack((numpy_data[0],numpy_data[1]))

    new_label = []
    for label in numpy_data[:,0]:
        new_label.append(label.replace('away ',''))

    sum_array = np.add(numpy_data[:,1],numpy_data[:,3])

    sorted_importance_index = sum_array.argsort()

    plt.barh(np.array(new_label)[sorted_importance_index], np.array(sum_array)[sorted_importance_index])
    plt.show()