import matplotlib.pyplot as plt
import pandas as pd

import os

overlap = False

# directories = ["D:/Desktop/graphs/new system/topology/smote/"]
path = "D:/Desktop/graphs/new system/final/"
directories = [os.path.join(path, i) + '/' for i in os.listdir(path)]
# directories = [os.path.join(path,i)+'/' for i in ['0.2','normal']]
# directories = [path]
if overlap == False:
    labels = ['' for i in range(0, len(directories))]
if overlap == True:
    fig, (ax1, ax2) = plt.subplots(2)
    accuracy_fig = plt.figure()
    loss_fig = plt.figure()
    labels = ['0.2','normal']
    # labels = os.listdir(path)

for idx, directory in enumerate(directories):
    if os.path.isdir(directory):
        if overlap == False:
            fig, (ax1, ax2) = plt.subplots(2)
            accuracy_fig = plt.figure()
            loss_fig = plt.figure()

        train_path = os.path.join(directory, 'train', 'all_training_logs_in_one_file.csv').replace("//", "/")
        validation_path = os.path.join(directory, 'validation', 'all_training_logs_in_one_file.csv').replace("//", "/")

        print(train_path)
        print(validation_path)

        train_df = pd.read_csv(train_path)
        validation_df = pd.read_csv(validation_path)

        # plot Accuracy
        # color=next(plt.gca()._get_lines.prop_cycler)['color']
        train_df_accuracy = train_df[train_df['metric'] == 'epoch_accuracy']
        validation_df_accuracy = validation_df[validation_df['metric'] == 'epoch_accuracy']
        ax1.plot(train_df_accuracy['step'], train_df_accuracy['value'], label='%s train' % labels[idx])
        ax1.plot(validation_df_accuracy['step'], validation_df_accuracy['value'], label='%s validation' % labels[idx],
                 linestyle='dashed')

        accuracy_fig.gca().plot(train_df_accuracy['step'], train_df_accuracy['value'], label='%s train' % labels[idx])
        accuracy_fig.gca().plot(validation_df_accuracy['step'], validation_df_accuracy['value'],
                                label='%s validation' % labels[idx],
                                linestyle='dashed')
        # plot Loss
        # color=next(plt.gca()._get_lines.prop_cycler)['color']
        train_df_loss = train_df[train_df['metric'] == 'epoch_loss']
        validation_df_loss = validation_df[validation_df['metric'] == 'epoch_loss']
        ax2.plot(train_df_loss['step'], train_df_loss['value'], label='%s train' % labels[idx])
        ax2.plot(validation_df_loss['step'], validation_df_loss['value'], label='%s validation' % labels[idx],
                 linestyle='dashed')
        loss_fig.gca().plot(train_df_loss['step'], train_df_loss['value'], label='%s train' % labels[idx])
        loss_fig.gca().plot(validation_df_loss['step'], validation_df_loss['value'],
                            label='%s validation' % labels[idx],
                            linestyle='dashed')
        if overlap == False:
            # plt.figure(0)
            # plt.legend()
            # plt.figure(1)
            # plt.legend()
            #
            # plt.figure(0)
            #
            #
            # plt.savefig(os.path.join(directory, os.path.basename(directory[:-1]) + 'accuracy.png'))
            # plt.figure(1)
            # plt.savefig(os.path.join(directory, os.path.basename(directory[:-1]) + 'loss.png'))

            ax1.legend()
            ax2.legend()
            accuracy_fig.gca().legend()
            loss_fig.gca().legend()

            ax1.set_title('Training and Validation Accuracy')
            ax2.set_title('Training and Validation Loss')
            fig.tight_layout()
            fig.savefig(os.path.join(directory, os.path.basename(directory[:-1]) + 'Both.png'))

            accuracy_fig.savefig(os.path.join(directory, os.path.basename(directory[:-1]) + 'accuracy.png'))

            loss_fig.savefig(os.path.join(directory, os.path.basename(directory[:-1]) + 'loss.png'))

            fig.show()
            accuracy_fig.show()
            loss_fig.show()
            plt.show()

if overlap == True:
    ax1.legend()
    ax2.legend()
    accuracy_fig.gca().legend()
    loss_fig.gca().legend()

    ax1.set_title('Training and Validation Accuracy')
    ax2.set_title('Training and Validation Loss')
    fig.tight_layout()
    fig.savefig(os.path.join(directory, '..',os.path.basename(directory[:-1]) + 'Both.png'))

    accuracy_fig.savefig(os.path.join(directory, '..',os.path.basename(directory[:-1]) + 'accuracy.png'))

    loss_fig.savefig(os.path.join(directory, '..',os.path.basename(directory[:-1]) + 'loss.png'))

    fig.show()
    accuracy_fig.show()
    loss_fig.show()
