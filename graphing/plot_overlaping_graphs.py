import matplotlib.pyplot as plt
import pandas as pd

import os


directories = ['D:/Desktop/graphs/final final/topology/ratings/%d/' % i for i in [1,6,7]]
print(directories)
labels= ['class weights','oversample','SMOTE']

for i in range(0,len(directories)):
    directory = directories[i]
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            print(filename)
            if 'accuracy' in filename:
                if 'train' in filename:
                    df = pd.read_csv(directory + filename)
                    plt.figure(0)
                    plt.plot(df['Step'], df['Value'], label='%s train' % labels[i])
                if 'validation' in filename:
                    df = pd.read_csv(directory + filename)
                    plt.figure(0)
                    plt.plot(df['Step'], df['Value'], label='%s validation' % labels[i],linestyle='dashed')
            if 'loss' in filename:
                if 'train' in filename:
                    df = pd.read_csv(directory + filename)
                    plt.figure(1)
                    plt.plot(df['Step'], df['Value'], label='%s train' % labels[i])
                if 'validation' in filename:
                    df = pd.read_csv(directory + filename)
                    plt.figure(1)
                    plt.plot(df['Step'], df['Value'], label='%s validation' % labels[i],linestyle='dashed' )



plt.figure(0)
plt.legend()
plt.figure(1)
plt.legend()

plt.figure(0)
plt.savefig(directory+'../'+'accuracy.png')
plt.figure(1)
plt.savefig(directory+'../'+'loss.png')

plt.show()

