import json
import cv2
import numpy as np
import os
import pickle
import glob
from random import shuffle

def create_dataset(split_file_name, split_name):
    dataset = []
    loc_from = '/home/preetham/Documents/Preetham/masters-thesis/data/sign-to-gloss/original/'
    with open(loc_from + 'split-files/' + split_file_name, 'r') as f:
        d = json.load(f)
    f.close()
    videos_skipped = 0
    c = 0
    for i in d.keys():
        if d[i]['subset'] != split_name:
            continue
        video_path = loc_from + 'videos/' + i + '.mp4'
        if not os.path.exists(video_path):
            continue
        n_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
        if n_frames - 0 < 9:
            videos_skipped += 1
            continue
        labels = d[i]['action'][0]
        if len(i) == 5:
            dataset.append((i, labels, 0, d[i]['action'][2]-d[i]['action'][1]))
        elif len(i) == 6:
            dataset.append((i, labels, 0, d[i]['action'][1], d[i]['action'][2] - d[i]['action'][1]))
        c += 1
    return dataset

def retrieve_available_num_classes(split_file_name):
    classes = []
    loc_from = '/home/preetham/Documents/Preetham/masters-thesis/data/sign-to-gloss/original/split-files/'
    with open(loc_from + split_file_name, 'r') as f:
        d = json.load(f)
    f.close()
    for i in d:
        classes.append(d[i]['action'][0])
    return np.unique(classes)

def create_sentence_dataset(dataset, n_size, n_words):
    new_dataset = []
    classes = []
    c = 0
    d = {}
    for i in range(n_size):
        row = []
        row_class = []
        f = True
        while f:
            ind = np.random.randint(0, len(dataset))
            word = dataset[ind]
            if len(row_class) < n_words:
                if word[1] not in row_class:
                    row_class.append(word[1])
                    row.append(int(word[0]))
                    d[int(word[0])] = word[1:]
            else:
                row_class.sort()
                if row_class in classes:
                    row = []
                    row_class = []
                    f = True
                else:
                    classes.append(row_class)
                    new_dataset.append(row)
                    f = False
        c += 1
    return new_dataset, d

def dataset_converter(phrases, n_classes):
    test_phrase = phrases[:2000]
    val_phrase = phrases[2000:4000]
    train_phrase = phrases[4000:]
    print('New Training set size: ', len(train_phrase))
    print('New Validation set size: ', len(val_phrase))
    print('New Testing set size: ', len(test_phrase))
    print()
    save_file(train_phrase, 'train-phrase-' + str(n_classes))
    save_file(val_phrase, 'val-phrase-' + str(n_classes))
    save_file(test_phrase, 'test-phrase-' + str(n_classes))

def save_file(d, name):
    loc_to = '/home/preetham/Documents/Preetham/masters-thesis/data/sign-to-gloss/cleaned/split-files/'
    with open(loc_to + name + '.pkl', 'wb') as f:
        pickle.dump(d, f, protocol=2)
    print(name + ' saved successfully')
    f.close()

def check_files(dataset):
    new_dataset = []
    loc = '/home/preetham/Documents/Preetham/masters-thesis/data/sign-to-gloss/cleaned/keypoints/'
    files = glob.glob(loc+'*.npy')
    for i in dataset:
        file_loc = loc + i[0] + '.npy'
        if file_loc in files:
            new_dataset.append(i)
    return new_dataset

def main():
    n_classes = 2000
    split_file_name = 'WLASL_' + str(n_classes) + '.json'
    classes = retrieve_available_num_classes(split_file_name)
    print()
    print('Number of Classes: ', len(classes))
    print()
    train_dataset = create_dataset(split_file_name, 'train')
    print('Original Training set size: ', len(train_dataset))
    val_dataset = create_dataset(split_file_name, 'val')
    print('Original Validation set size: ', len(val_dataset))
    test_dataset = create_dataset(split_file_name, 'test')
    print('Original Testing set size: ', len(test_dataset))
    print()
    dataset = train_dataset + val_dataset + test_dataset
    print('Original Total Dataset size: ', len(dataset))
    print()
    dataset = check_files(dataset)
    print('New Dataset size after checking files: ', len(dataset))
    print()
    """dataset_phrase, dataset_info = create_sentence_dataset(dataset, 254000, 4)
    print('New Phrase Dataset size: ', len(dataset_phrase))
    print()
    shuffle(dataset_phrase)
    save_file(dataset_info, 'dataset-info-' + str(n_classes))
    print()
    dataset_converter(dataset_phrase, n_classes)
    print()"""

main()