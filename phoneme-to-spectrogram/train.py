import tensorflow as tf
import os
import logging
from model import DecoderPreNet
from utils import text_retrieve, open_file, create_batch
from random import shuffle
#from utils import open_file, save_file, model_training, model_testing, text_retrieve, tokenize, create_new_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def main():
    print()
    loc = '/home/preetham/Documents/Preetham/masters-thesis/'
    files = text_retrieve('files_list.txt')
    print('No. of files in original dataset: ', len(files))
    print()
    shuffle(files)
    train, val, test = files[:1000], files[:20], files[:20]
    print('No. of files in training dataset: ', len(train))
    print('No. of files in validation dataset: ', len(val))
    print('No. of files in testing dataset: ', len(test))
    print()
    inp_word_index = open_file('results/grapheme-to-phoneme/luong/model_7/utils/tar-word-index.pkl')
    start_index = 0
    batch_size = 8
    train = train[start_index:start_index+batch_size]
    train_batch_inp, train_batch_tar = create_batch(train, inp_word_index)
    dec_pre_net = DecoderPreNet(256, 0.1)
    print(train_batch_tar.shape)
    print(train_batch_tar[:,:, 0].shape)
    x = dec_pre_net(train_batch_tar[:, 0], False)
    print(x.shape)

main()