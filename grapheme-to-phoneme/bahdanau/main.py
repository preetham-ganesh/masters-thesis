import os
import time

model_no = [1, 2, 3, 4, 5, 6, 7, 8]
print()
for i in model_no:
    loc = '/home/preetham/Documents/Preetham/masters-thesis/results/grapheme-to-phoneme'
    try:
        os.mkdir(loc + '/bahdanau/')
        print(loc + '/bahdanau folder created')
    except:
        print(loc + '/bahdanau folder exists')
    try:
        os.mkdir(loc + '/bahdanau/model_' + str(i))
        print(loc + '/bahdanau/model_' + str(i) + ' folder created')
    except:
        print(loc + '/bahdanau/model_' + str(i) + ' folder exists')
    try:
        os.mkdir(loc + '/bahdanau/model_' + str(i) + '/utils')
        print(loc + '/bahdanau/model_' + str(i) + '/utils folder created')
    except:
        print(loc + '/bahdanau/model_' + str(i) + '/utils folder exists')
    try:
        os.mkdir(loc + '/bahdanau/model_' + str(i) + '/predictions')
        print(loc + '/bahdanau/model_' + str(i) + '/predictions folder created')
    except:
        print(loc + '/bahdanau/model_' + str(i) + '/predictions folder exists')
    try:
        os.mkdir(loc + '/bahdanau/model_' + str(i) + '/history')
        print(loc + '/bahdanau/model_' + str(i) + '/history folder created')
    except:
        print(loc + '/bahdanau/model_' + str(i) + '/history folder exists')
    print()
    cmd_1 = 'python3 train.py ' + str(i)
    os.system(cmd_1)
    cmd_2 = 'python3 predict.py ' + str(i) + ' val'
    os.system(cmd_2)
    cmd_3 = 'python3 predict.py ' + str(i) + ' test'
    os.system(cmd_3)
    print()
