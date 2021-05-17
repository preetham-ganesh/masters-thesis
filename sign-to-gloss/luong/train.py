import tensorflow as tf
import os
import logging
from utils import open_file, save_file, create_batch, model_training, model_testing
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def main():
    print()
    n_classes = int(sys.argv[1])
    model = int(sys.argv[2])
    dataset_info = open_file('data/sign-to-gloss/cleaned/split-files/dataset-info-' + str(n_classes))
    print('Dataset Info set size: ', len(dataset_info.keys()))
    print()
    train_phrase = open_file('data/sign-to-gloss/cleaned/split-files/train-phrase-' + str(n_classes))
    val_phrase = open_file('data/sign-to-gloss/cleaned/split-files/val-phrase-' + str(n_classes))
    test_phrase = open_file('data/sign-to-gloss/cleaned/split-files/test-phrase-' + str(n_classes))
    print('Training Phrase set size: ', len(train_phrase))
    print('Validation Phrase set size: ', len(val_phrase))
    print('Testing Phrase set size: ', len(test_phrase))
    print()
    batch_size = 50
    vocab_size = n_classes + 2
    parameters = {'tar_vocab_size': vocab_size, 'emb_size': 512, 'rnn_size': 512, 'batch_size': batch_size,
                  'epochs': 20, 'train_steps_per_epoch': len(train_phrase) // batch_size, 'rate': 0.3,
                  'val_steps_per_epoch': len(val_phrase) // batch_size, 'test_steps': len(test_phrase) // batch_size,
                  'model': model}
    save_file(parameters, 'results/sign-to-gloss/wlasl-' + str(n_classes) + '/luong/model_' + str(model) + '/utils/parameters')
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
    #model_training(train_dataset, val_dataset, dataset_info, parameters)
    print('Model Testing started')
    print()
    model_testing(test_dataset, dataset_info, parameters)

main()