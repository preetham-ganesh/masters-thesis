import os
import logging
import tensorflow as tf
from utils import translate, open_file, create_batch
import matplotlib.pyplot as plt
import numpy as np
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def text_save(text, name):
    f = open('/home/preetham/Documents/Preetham/masters-thesis/results/sign-to-gloss/wlasl-' + name, 'w', encoding='utf-8')
    f.write(text)
    f.close()

def lines_to_text(lines, sep):
    text = ''
    for i in range(len(lines)):
        if i == len(lines) - 1:
            text += str(lines[i])
        else:
            text += str(lines[i]) + sep
    return text

def text_retrieve(name):
    with open('/home/preetham/Documents/Preetham/masters-thesis/data/sign-to-gloss/original/'+name, 'r',
              encoding='utf-8') as f:
        text = f.read()
    f.close()
    return text.split('\n')

def convert_tar_pred(tar, pred, class_list):
    tar = [class_list[i].split('\t')[1] for i in tar]
    pred = [class_list[i].split('\t')[1] for i in pred]
    tar = ' '.join(tar)
    pred = ' '.join(pred)
    return tar, pred

def main():
    print()
    model = int(sys.argv[1])
    file_name = sys.argv[2]
    n_classes = int(sys.argv[3])
    class_list = text_retrieve('class_list.txt')
    val_phrase = open_file('data/sign-to-gloss/cleaned/split-files/'+file_name+'-phrase-'+str(n_classes))
    dataset_info = open_file('data/sign-to-gloss/cleaned/split-files/dataset-info-'+str(n_classes))
    tar_lines, pred_lines = [], []
    for i in range(0, len(val_phrase)):
        print(i)
        inp, tar = create_batch([val_phrase[i]], dataset_info, n_classes)
        pred = translate(inp, model, n_classes)
        tar, pred = convert_tar_pred(list(tar[0][1:-1]), pred, class_list)
        print('Target phrase: ', tar)
        print('Predict phrase: ', pred)
        print()
        tar_lines.append(tar)
        pred_lines.append(pred)
    tar_text = lines_to_text(tar_lines, '\n')
    pred_text = lines_to_text(pred_lines, '\n')
    text_save(tar_text, str(n_classes) + '/luong/model_' + str(model) + '/predictions/' + file_name + '_tar.txt')
    text_save(pred_text, str(n_classes) + '/luong/model_' + str(model) + '/predictions/' + file_name + '_pred.txt')

main()