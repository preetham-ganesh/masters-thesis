import os
import logging
import tensorflow as tf
from utils import translate, text_retrieve
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def preprocess_inp_tar(word):
    word_list = word.split(' ')
    return ' '.join(word_list[1:-1])
    
def main():
    inp = 'v a s u'
    model = 7
    print(translate(inp, model))

main()
