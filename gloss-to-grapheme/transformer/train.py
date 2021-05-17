import tensorflow as tf
import os
import logging
from utils import open_file, save_file, model_training, model_testing, text_retrieve, tokenize, create_new_dataset
import tensorflow_datasets as tfds
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def main():
    print()
    model = int(sys.argv[1])
    train_inp = open_file('data/gloss-to-grapheme/swt-tokenized/train.gloss')
    val_inp = open_file('data/gloss-to-grapheme/swt-tokenized/val.gloss')
    test_inp = open_file('data/gloss-to-grapheme/swt-tokenized/test.gloss')
    train_tar = open_file('data/gloss-to-grapheme/swt-tokenized/train.en')
    val_tar = open_file('data/gloss-to-grapheme/swt-tokenized/val.en')
    test_tar = open_file('data/gloss-to-grapheme/swt-tokenized/test.en')
    print('No. of original sentences in Training set: ', len(train_inp))
    print('No. of original sentences in Validation set: ', len(val_inp))
    print('No. of original sentences in Test set: ', len(test_inp))
    print()
    train_inp, val_inp, test_inp = tokenize(train_inp, val_inp, test_inp)
    train_tar, val_tar, test_tar = tokenize(train_tar, val_tar, test_tar)
    batch_size = 128
    loc_from = '/home/preetham/Documents/Preetham/masters-thesis/results/gloss-to-grapheme/tokenizer/'
    inp_lang = tfds.deprecated.text.SubwordTextEncoder.load_from_file(loc_from + 'gloss-swt')
    tar_lang = tfds.deprecated.text.SubwordTextEncoder.load_from_file(loc_from + 'en-swt')
    print('Input Vocabulary size: ', inp_lang.vocab_size + 2)
    print('Target Vocabulary size: ', tar_lang.vocab_size + 2)
    print()
    if model <= 4:
    	n_layers = model
    	d_model = 512
    	dropout = 0.1
    	n_heads = 8
    else:
    	n_layers = model - 4
    	d_model = 1024
    	dropout = 0.3
    	n_heads = 16
    parameters = {'inp_vocab_size': inp_lang.vocab_size + 2, 'tar_vocab_size': tar_lang.vocab_size + 2,
                  'n_layers': n_layers, 'd_model': d_model, 'dff': 4*d_model, 'batch_size': batch_size, 'epochs': 30,
                  'n_heads': n_heads, 'train_steps_per_epoch': len(train_inp) // batch_size, 'dropout': dropout,
                  'val_steps_per_epoch': len(val_inp) // batch_size, 'test_steps': len(test_inp) // batch_size,
                  'model': model}
    save_file(parameters, 'results/gloss-to-grapheme/transformer/model_'+str(model)+'/utils/parameters')
    print()
    print('No. of Training steps per epoch: ', parameters['train_steps_per_epoch'])
    print('No. of Validation steps per epoch: ', parameters['val_steps_per_epoch'])
    print('No. of Testing steps: ', parameters['test_steps'])
    print()
    train_dataset = tf.data.Dataset.from_tensor_slices((train_inp, train_tar))
    train_dataset = train_dataset.shuffle(len(train_inp)).padded_batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_inp, val_tar))
    val_dataset = val_dataset.shuffle(len(val_inp)).padded_batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_inp, test_tar))
    test_dataset = test_dataset.shuffle(len(test_inp)).padded_batch(batch_size)
    model_training(train_dataset, val_dataset, parameters)
    model_testing(test_dataset, parameters)

main()
