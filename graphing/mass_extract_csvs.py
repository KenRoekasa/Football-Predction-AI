import os
from graphing.extract_csvs import main
path = "D:/Desktop/kxr758/logs/final_final/new system/absolute_final/dropout/"


for dates in os.listdir(path):
    dates_folder = os.path.join(path, dates)
    for f in os.listdir(dates_folder):
        train_path = os.path.join(dates_folder, f, 'train').replace("//","/")
        validation_path = os.path.join(dates_folder, f, 'validation').replace("//","/")

        # print(train_path)
        # print(validation_path)

        main(logdir_or_logfile=train_path, out_dir=train_path,write_pkl=False,write_csv=True)
        main(logdir_or_logfile=validation_path, out_dir=validation_path,write_pkl=False,write_csv=True)

