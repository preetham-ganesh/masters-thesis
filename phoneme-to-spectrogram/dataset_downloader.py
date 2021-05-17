import tensorflow_datasets as tfds
import os
import logging
import pickle
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

def convert_tensor_to_lines(tensor, name):
    c = 0
    for i in tensor:
        try:
            id = str(i['id'].decode('utf-8'))
            x = str(i['text_normalized'].decode('utf-8'))
            y = np.array(tf.cast(i['speech'], tf.float32).numpy())
            d = {'id': id, 'text_normalized': x, 'speech': y}
            dataset_save(d, id)
            c += 1
        except:
            continue
    print()
    print('No. of files after conversion: ', c)
    print()

def dataset_save(d, name):
    loc_to = '/home/preetham/Documents/Preetham/masters-thesis/data/phoneme-to-spectrogram/original/'
    with open(loc_to + name + '.pkl', 'wb') as f:
        pickle.dump(d, f, protocol=2)
    print(name + ' saved successfully')
    f.close()

def main():
    print()
    train = tfds.as_numpy(dataset = tfds.load('ljspeech', split='train', download=True, shuffle_files=True))
    convert_tensor_to_lines(train, 'dataset')

main()
