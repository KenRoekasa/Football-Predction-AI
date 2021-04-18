import matplotlib.pyplot as plt
import pandas as pd

import os

directory = "D:/Desktop/graphs/learning rate/batch/16/"
for filename in os.listdir(directory):
    if 'accuracy' in filename:
        if 'train' in filename:
            df = pd.read_csv(directory + filename)
            plt.figure(0)
            plt.plot(df['Step'], df['Value'], label='train')
        if 'validation' in filename:
            df = pd.read_csv(directory + filename)
            plt.figure(0)
            plt.plot(df['Step'], df['Value'], label='validation')
    if 'loss' in filename:
        if 'train' in filename:
            df = pd.read_csv(directory + filename)
            plt.figure(1)
            plt.plot(df['Step'], df['Value'], label='train')
        if 'validation' in filename:
            df = pd.read_csv(directory + filename)
            plt.figure(1)
            plt.plot(df['Step'], df['Value'], label='validation')


plt.figure(0)
plt.legend()
plt.figure(1)
plt.legend()

plt.figure(0)
plt.savefig(directory+'16-accuracy.png')
plt.figure(1)
plt.savefig(directory+'16-loss.png')

plt.show()

