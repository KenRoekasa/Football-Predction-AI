import matplotlib.pyplot as plt
import pandas as pd

import os

directories = ['D:/Desktop/graphs/final final/topology/order/%d/' % i for i in range(1,11)]
print(directories)
labels = [str(i) for i in range(1,11)]

for idx, directory in enumerate(directories):
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            print(filename)
            if 'accuracy' in filename:
                if 'train' in filename:
                    df = pd.read_csv(directory + filename)
                    plt.figure(0)
                    plt.plot(df['Step'], df['Value'], label='train')
                if 'validation' in filename:
                    df = pd.read_csv(directory + filename)
                    plt.figure(0)
                    plt.plot(df['Step'], df['Value'], label='validation', linestyle='dashed')
            if 'loss' in filename:
                if 'train' in filename:
                    df = pd.read_csv(directory + filename)
                    plt.figure(1)
                    plt.plot(df['Step'], df['Value'], label='train')
                if 'validation' in filename:
                    df = pd.read_csv(directory + filename)
                    plt.figure(1)
                    plt.plot(df['Step'], df['Value'], label='validation',linestyle='dashed')

    plt.figure(0)
    plt.legend()
    plt.figure(1)
    plt.legend()

    plt.figure(0)
    plt.savefig(directory + str(labels[idx]) + 'accuracy.png')
    plt.figure(1)
    plt.savefig(directory + str(labels[idx]) + 'loss.png')

    plt.show()
