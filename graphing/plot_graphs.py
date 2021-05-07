import matplotlib.pyplot as plt
import pandas as pd

import os


path = "D:/Desktop/graphs/final final/resample/"
directories = [os.path.join(path,i)+'/' for i in os.listdir(path)]

# labels = [str(i) for i in range(1,11)]
labels = ['' for i in range(0,len(directories))]

for idx, directory in enumerate(directories):
    fig, (ax1, ax2) = plt.subplots(2)
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            print(filename)
            if 'accuracy' in filename:
                if 'train' in filename:
                    df = pd.read_csv(directory + filename)

                    ax1.plot(df['Step'], df['Value'], label='train')
                if 'validation' in filename:
                    df = pd.read_csv(directory + filename)
                    ax1.plot(df['Step'], df['Value'], label='validation', linestyle='dashed')
            if 'loss' in filename:
                if 'train' in filename:
                    df = pd.read_csv(directory + filename)

                    ax2.plot(df['Step'], df['Value'], label='train')
                if 'validation' in filename:
                    df = pd.read_csv(directory + filename)

                    ax2.plot(df['Step'], df['Value'], label='validation',linestyle='dashed')

    ax1.legend()
    ax2.legend()

    ax1.set_title('Training and Validation Accuracy')
    ax2.set_title('Training and Validation Loss')
    fig.tight_layout()
    fig.savefig(os.path.join(directory, os.path.basename(directory[:-1]) + 'Both.png'))
    fig.show()
