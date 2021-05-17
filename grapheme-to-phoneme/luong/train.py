import tensorflow as tf
import os
import logging
from utils import tokenize, save_file, text_retrieve, model_training, model_testing, create_new_dataset
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def main():
    print()
    model = int(sys.argv[1])
    train_inp = text_retrieve('train.en')
    val_inp = text_retrieve('val.en')
    test_inp = text_retrieve('test.en')
    train_tar = text_retrieve('train.phone')
    val_tar = text_retrieve('val.phone')
    test_tar = text_retrieve('test.phone')
    print('No. of original sentences in Training set: ', len(train_inp))
    print('No. of original sentences in Validation set: ', len(val_inp))
    print('No. of original sentences in Test set: ', len(test_inp))
    print()
    max_length = 25
    train_inp, train_tar = create_new_dataset(train_inp, train_tar, max_length)
    val_inp, val_tar = create_new_dataset(val_inp, val_tar, max_length)
    test_inp, test_tar = create_new_dataset(test_inp, test_tar, max_length)
    print('No. of new sentences in Training set: ', len(train_inp))
    print('No. of new sentences in Validation set: ', len(val_inp))
    print('No. of new sentences in Test set: ', len(test_inp))
    print()
    inp_lang, train_inp, val_inp, test_inp = tokenize(train_inp, val_inp, test_inp, max_length)
    tar_lang, train_tar, val_tar, test_tar = tokenize(train_tar, val_tar, test_tar, max_length)
    print('Input Vocabulary size: ', len(inp_lang.word_index) + 1)
    print('Target Vocabulary size: ', len(tar_lang.word_index) + 1)
    print()
    batch_size = 128
    save_file(inp_lang.word_index, 'model_' + str(model) + '/utils/inp-word-index')
    save_file(inp_lang.index_word, 'model_' + str(model) + '/utils/inp-index-word')
    save_file(tar_lang.word_index, 'model_' + str(model) + '/utils/tar-word-index')
    save_file(tar_lang.index_word, 'model_' + str(model) + '/utils/tar-index-word')
    parameters = {'inp_vocab_size': len(inp_lang.word_index) + 1, 'tar_vocab_size': len(tar_lang.word_index) + 1,
                  'emb_size': 512, 'rnn_size': 512, 'batch_size': batch_size, 'epochs': 20,
                  'train_steps_per_epoch': len(train_inp) // batch_size, 'rate': 0.3,
                  'val_steps_per_epoch': len(val_inp) // batch_size, 'test_steps': len(test_inp) // batch_size,
                  'max_length': max_length, 'model': model}
    save_file(parameters, 'model_' + str(model) + '/utils/parameters')
    print()
    print('No. of Training steps per epoch: ', parameters['train_steps_per_epoch'])
    print('No. of Validation steps per epoch: ', parameters['val_steps_per_epoch'])
    print('No. of Testing steps: ', parameters['test_steps'])
    print()
    train_dataset = tf.data.Dataset.from_tensor_slices((train_inp, train_tar)).shuffle(len(train_inp))
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_inp, val_tar)).shuffle(len(val_inp))
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_inp, test_tar)).shuffle(len(test_inp))
    test_dataset = test_dataset.batch(batch_size, drop_remainder=True)
    print('Model training started')
    print()
    model_training(train_dataset, val_dataset, parameters)
    model_testing(test_dataset, parameters)

main()
