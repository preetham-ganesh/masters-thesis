import os
import logging
import tensorflow as tf
from utils import translate, open_file, create_batch
import matplotlib.pyplot as plt
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(physical_devices[1], 'GPU')
tf.config.experimental.set_memory_growth(physical_devices[1], enable=True)

def text_save(text, name):
    f = open('/home/ganesh/cse-5698/results/sign-to-gloss/bahdanau/' + name, 'w', encoding='utf-8')
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

def keypoint_rearrange(keypoints):
    head = [list(keypoints[0])] + [list(i) for i in keypoints[14:18]] + [list(keypoints[1])]
    right_shoulder_hand = [list(i) for i in keypoints[2:4]] + [list(i) for i in keypoints[39:]]
    left_shoulder_hand = [list(i) for i in keypoints[5:7]] + [list(i) for i in keypoints[18:39]]
    new_keypoints = head + right_shoulder_hand + left_shoulder_hand
    return new_keypoints

def main():
    model = 1
    file_name = 'train'
    n_classes = 100
    val_phrase = open_file('data/sign-to-gloss/cleaned/split-files/'+file_name+'-phrase-'+str(n_classes))
    val_info = open_file('data/sign-to-gloss/cleaned/split-files/'+file_name+'-info-'+str(n_classes))
    inp_lines, tar_lines, pred_lines = [], [], []
    for i in range(10, 11):
        inp, tar = create_batch([val_phrase[i]], val_info, n_classes)
        translate(inp, model)
        print(tar)
        """print('Input sentence: ', preprocess_inp_tar(inp))
        print('Target sentence: ', preprocess_inp_tar(tar))
        print('Predict sentence: ', pred)
        print()
        inp_lines.append(preprocess_inp_tar(inp))
        tar_lines.append(preprocess_inp_tar(tar))
        pred_lines.append(pred)
    inp_text = lines_to_text(inp_lines, '\n')
    tar_text = lines_to_text(tar_lines, '\n')
    pred_text = lines_to_text(pred_lines, '\n')
    text_save(inp_text, 'model_' + str(model) + '/predictions/' + file_name + '_inp.txt')
    text_save(tar_text, 'model_' + str(model) + '/predictions/' + file_name + '_tar.txt')
    text_save(pred_text, 'model_' + str(model) + '/predictions/' + file_name + '_pred.txt')"""

main()