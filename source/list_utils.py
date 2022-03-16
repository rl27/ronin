import os
import numpy as np
import re

# Returns 1 or 2, for train_dataset_1 or train_dataset_2
def find_dataset(dirname, folder):
    for item in os.listdir(dirname + "train_dataset_1"):
        if item == folder:
            return 1
    for item in os.listdir(dirname + "train_dataset_2"):
        if item == folder:
            return 2
    return 0

# Generate train and val from train log
def gen_from_log(dirname, logname):
    f = open(logname, "r")
    count = 0
    train_str = ""
    train_ct = 0
    val_str = ""
    val_ct = 0
    train = True
    for line in f:
        if re.search("^a[0-9]{3}_[0-9]", line):
            folder = line[:6]
            d = find_dataset(dirname, folder)
            if train:
                train_str += "train_dataset_{}/{}\n".format(d, folder)
                train_ct += 1
            else:
                val_str += "train_dataset_{}/{}\n".format(d, folder)
                val_ct += 1
            count += 1
        elif re.search("^Training set loaded", line):
            train = False

    f = open("train_list.txt", "w")
    f.write(train_str)
    f.close()
    f = open("val_list.txt", "w")
    f.write(val_str)
    f.close()
    print("Train items:", train_ct)
    print("Val items:", val_ct)
    print("Total:", count)

# Generate list
def gen_list(dirname):
    testlist = ""
    for item in os.listdir(dirname):
        testlist += item + '\n'
    f = open("my_list.txt", "w")
    #f.write("a047_2")
    f.write(testlist)
    f.close()

# Generate train list, given dataset directory
def gen_train_list(dirname):
    items = ""
    for item in os.listdir(dirname + "train_dataset_1"):
        items += "train_dataset_1/" + item + '\n'
    for item in os.listdir(dirname + "train_dataset_2"):
        items += "train_dataset_2/" + item + '\n'
    f = open("train_list.txt", "w")
    f.write(items)
    f.close()

# Generate train and val lists. 70/85 is train, 15/85 is val. Randomly selected.
def gen_train_val(dirname, dir2):
    items = []
    for item in os.listdir(dirname + "train_dataset_1"):
        items.append("train_dataset_1/" + item + '\n')
    for item in os.listdir(dirname + "train_dataset_2"):
        items.append("train_dataset_2/" + item + '\n')
    train = np.random.choice(items, 70, replace=False)  # 70 out of 85 is train
    val = []
    for i in items:
        if i not in train:
            val.append(i)  # 15 out of 85 is val
    train_str = ""
    for i in train:
        train_str += i
    val_str = ""
    for i in val:
        val_str += i
    f = open("train_list.txt", "w")
    f.write(train_str)
    f.close()
    f = open("val_list.txt", "w")
    f.write(val_str)
    f.close()
    
    if not os.path.exists(dir2):
        os.makedirs(dir2)
    
    f = open(dir2 + "/train_list.txt", "w")
    f.write(train_str)
    f.close()
    f = open(dir2 + "/val_list.txt", "w")
    f.write(val_str)
    f.close()