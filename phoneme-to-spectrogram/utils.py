import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

def text_retrieve(name):
    with open('/home/preetham/Documents/Preetham/masters-thesis/data/phoneme-to-spectrogram/cleaned/'+name, 'r',
              encoding='utf-8') as f:
        text = f.read()
    f.close()
    return text.split('\n')

def open_file(name):
    loc_to = '/home/preetham/Documents/Preetham/masters-thesis/'
    with open(loc_to + name, 'rb') as f:
        d = pickle.load(f)
    f.close()
    return d

def create_batch(files, inp_word_index):
    inp, tar = [], []
    max_len = 0
    for i in files:
        d = open_file('data/phoneme-to-spectrogram/cleaned/' + i)
        print(d['phoneme'])
        phoneme = d['phoneme'].replace('  ', ' ')
        phoneme = [inp_word_index[j] for j in phoneme.split(' ')]
        if max_len < d['mel_spectrogram'].shape[1]:
            max_len = d['mel_spectrogram'].shape[1]
        inp.append(phoneme)
        tar.append(d['mel_spectrogram'])
    new_tar = []
    for i in tar:
        t = tf.cast(i, tf.int64)
        t = tf.keras.preprocessing.sequence.pad_sequences(i, padding='post', maxlen=max_len)
        new_tar.append(t)
    inp = tf.keras.preprocessing.sequence.pad_sequences(inp, padding='post')
    tar = tf.keras.preprocessing.sequence.pad_sequences(new_tar, padding='post')
    inp = np.array(inp)
    tar = np.array(tar)
    return inp, tar