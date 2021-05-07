import os


path = "D:/Desktop/graphs/new system/regularisation/all-100-smote/dropout"


labels = [str(i/10) for i in range(1,10)]

for i in range(0,len(labels)):
    sub_path= os.path.join(path, labels[i]).replace("\\", "/")

    os.rename(os.path.join(sub_path,'accuracy.png').replace("\\", "/"),os.path.join(sub_path,'%saccuracy.png' % labels[i]).replace("\\", "/"))
    os.rename(os.path.join(sub_path,'loss.png').replace("\\", "/"),os.path.join(sub_path,'%sloss.png' % labels[i]).replace("\\", "/"))