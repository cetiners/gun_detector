import os
import shutil

# çiçek gibi kod


def sorter(train_size, test_size, valid_size):
    # iterate over the files in the folders, split depending on the pre-set tresholds 
    for cat in ["others","pistols"]:
        for filecount, filename in enumerate(os.listdir("data/raw/"+cat)):
            # if the sum of desired is more than number of pics, this will break
            if filecount < train_size/2:
                src = os.path.join("data/raw/"+cat,filename)
                dst = os.path.join("data/cooked/train/"+cat, filename)
                shutil.copyfile(src,dst);
            elif filecount < train_size/2 + test_size/2:
                src = os.path.join("data/raw/"+cat,filename)
                dst = os.path.join("data/cooked/test/"+cat, filename)
                shutil.copyfile(src,dst);
            elif filecount < train_size/2 + test_size/2 + valid_size/2:
                src = os.path.join("data/raw/"+cat,filename)
                dst = os.path.join("data/cooked/validation/"+cat, filename)
                shutil.copyfile(src,dst);
            else:
                break;


def cleaner(folder_name):
    for cat in ["others","pistols"]:
        # delete the files on the cooked folder before changing the split sizes to avoid duplication & ease of upload
        for file in os.listdir("data/cooked/"+folder_name+"/"+cat):
            os.remove(os.path.join("data/cooked/"+folder_name+"/"+cat, file))

def pürpak():
    [cleaner(i) for i in ["train","test","validation"]];