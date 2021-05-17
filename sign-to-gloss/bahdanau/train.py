import tensorflow as tf
import os
import logging
from utils import open_file, save_file, create_batch, model_training, model_testing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(physical_devices[1], 'GPU')
tf.config.experimental.set_memory_growth(physical_devices[1], enable=True)

def main():
    print()
    n_classes = 100
    model = 1
    train_info = open_file('data/sign-to-gloss/cleaned/split-files/train-info-'+str(100))
    val_info = open_file('data/sign-to-gloss/cleaned/split-files/val-info-'+str(100))
    test_info = open_file('data/sign-to-gloss/cleaned/split-files/test-info-'+str(100))
    print('Training Info set size: ', len(train_info.keys()))
    print('Validation Info set size: ', len(val_info.keys()))
    print('Testing Info set size: ', len(test_info.keys()))
    print()
    train_phrase = open_file('data/sign-to-gloss/cleaned/split-files/train-phrase-' + str(100))
    val_phrase = open_file('data/sign-to-gloss/cleaned/split-files/val-phrase-' + str(100))
    test_phrase = open_file('data/sign-to-gloss/cleaned/split-files/test-phrase-' + str(100))
    print('Training Phrase set size: ', len(train_phrase))
    print('Validation Phrase set size: ', len(val_phrase))
    print('Testing Phrase set size: ', len(test_phrase))
    print()
    batch_size = 50
    vocab_size = n_classes + 2
    parameters = {'tar_vocab_size': vocab_size, 'emb_size': 256, 'rnn_size': 256, 'batch_size': batch_size,
                  'epochs': 20, 'train_steps_per_epoch': len(train_phrase) // batch_size, 'rate': 0.1,
                  'val_steps_per_epoch': len(val_phrase) // batch_size, 'test_steps': len(test_phrase) // batch_size,
                  'model': model}
    save_file(parameters, 'results/sign-to-gloss/bahdanau/model_' + str(model) + '/utils/parameters')
    print()
    print('No. of Training steps per epoch: ', parameters['train_steps_per_epoch'])
    print('No. of Validation steps per epoch: ', parameters['val_steps_per_epoch'])
    print('No. of Testing steps: ', parameters['test_steps'])
    print()
    train_dataset = tf.data.Dataset.from_tensor_slices((train_phrase)).shuffle(len(train_phrase))
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_phrase)).shuffle(len(val_phrase))
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_phrase)).shuffle(len(test_phrase))
    test_dataset = test_dataset.batch(batch_size, drop_remainder=True)
    print('Model Training started')
    print()
    model_training(train_dataset, train_info, val_dataset, val_info, parameters)
    model_testing(test_dataset, test_info, parameters)

main()