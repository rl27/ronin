import os
import numpy as np

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
    
    f = open(dir2 + "/train_list.txt", "w")
    f.write(train_str)
    f.close()
    f = open(dir2 + "/val_list.txt", "w")
    f.write(val_str)
    f.close()