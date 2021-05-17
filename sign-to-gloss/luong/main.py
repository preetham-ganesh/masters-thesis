import os
import time

n_classes = 2000
model_no = [6, 7, 8]
print()
for i in model_no:
    loc = '/home/preetham/Documents/Preetham/masters-thesis/results/sign-to-gloss/'
    try:
        os.mkdir(loc + 'wlasl-' + str(n_classes) + '/luong/')
        print(loc + 'wlasl-' + str(n_classes) + '/luong folder created')
    except:
        print(loc + 'wlasl-' + str(n_classes) + '/luong folder exists')
    try:
        os.mkdir(loc + 'wlasl-' + str(n_classes) + '/luong/model_' + str(i))
        print(loc + 'wlasl-' + str(n_classes) + '/luong/model_' + str(i) + ' folder created')
    except:
        print(loc + 'wlasl-' + str(n_classes) + '/luong/model_' + str(i) + ' folder exists')
    try:
        os.mkdir(loc + 'wlasl-' + str(n_classes) + '/luong/model_' + str(i) + '/utils')
        print(loc + 'wlasl-' + str(n_classes) + '/luong/model_' + str(i) + '/utils folder created')
    except:
        print(loc + 'wlasl-' + str(n_classes) + '/luong/model_' + str(i) + '/utils folder exists')
    try:
        os.mkdir(loc + 'wlasl-' + str(n_classes) + '/luong/model_' + str(i) + '/predictions')
        print(loc + 'wlasl-' + str(n_classes) + '/luong/model_' + str(i) + '/predictions folder created')
    except:
        print(loc + 'wlasl-' + str(n_classes) + '/luong/model_' + str(i) + '/predictions folder exists')
    try:
        os.mkdir(loc + 'wlasl-' + str(n_classes) + '/luong/model_' + str(i) + '/history')
        print(loc + 'wlasl-' + str(n_classes) + '/luong/model_' + str(i) + '/history folder created')
    except:
        print(loc + 'wlasl-' + str(n_classes) + '/luong/model_' + str(i) + '/history folder exists')
    print()
    cmd_1 = 'python3 train.py ' + str(n_classes) + ' ' + str(i)
    os.system(cmd_1)
    cmd_2 = 'python3 predict.py ' + str(i) + ' val ' + str(n_classes)
    os.system(cmd_2)
    cmd_3 = 'python3 predict.py ' + str(i) + ' test ' + str(n_classes)
    os.system(cmd_2)
    print()