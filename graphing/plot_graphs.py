import matplotlib.pyplot as plt
import pandas as pd

import os

for i in ['0.1','0.01','0.001','0.0001','1e-05','1e-06']:
    directory = "D:/Desktop/learning rate/sgd/%s/" % i
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
                plt.figure(1)soAdd
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
    plt.savefig(directory+str(i)+'accuracy.png')
    plt.figure(1)
    plt.savefig(directory+str(i)+'loss.png')

    plt.show()

