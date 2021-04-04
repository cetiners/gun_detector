import os
import shutil
import random

def sorter(train_size, test_size, valid_size):
    # iterate over the files in the folders, split depending on the pre-set tresholds
    # Altough, they are called as "size", they are in fact "ratio"s! 
    for cat in ["pistols", "knives"]:
        totalcount = len(os.listdir("data/raw/knives"))
        for filename in os.listdir("data/raw/"+cat):
            picker = random.randint(1,3)
            # Use picker to randomly assign the images to the folders
            if (len(os.listdir("data/cooked/test/"+cat)) < test_size*totalcount) & (picker == 1):
                src = os.path.join("data/raw/"+cat,filename)
                dst = os.path.join("data/cooked/test/"+cat, filename)
                shutil.copyfile(src,dst)

            elif (len(os.listdir("data/cooked/validation/"+cat)) < valid_size*totalcount) & (picker <= 2):
                src = os.path.join("data/raw/"+cat,filename)
                dst = os.path.join("data/cooked/validation/"+cat, filename)
                shutil.copyfile(src,dst)

            elif (len(os.listdir("data/cooked/train/"+cat)) < train_size*totalcount) & (picker <= 3):
                src = os.path.join("data/raw/"+cat,filename)
                dst = os.path.join("data/cooked/train/"+cat, filename)
                shutil.copyfile(src,dst)

    for i in ["train", "test", "validation"]:
        for j in ["knives", "pistols"]:
            print('Total number of "cooked" images in', i +"/"+ j)
            print(len(os.listdir("data/cooked/" + i +"/"+ j)))

def cleaner(folder_name):
    for cat in ["knives","pistols"]:
        # delete the files on the cooked folder before changing the split sizes to avoid duplication & ease of upload
        for file in os.listdir("data/cooked/"+folder_name+"/"+cat):
            os.remove(os.path.join("data/cooked/"+folder_name+"/"+cat, file))
