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

def text_save(text, name):
    f = open('/home/preetham/Documents/Preetham/masters-thesis/results/grapheme-to-phoneme/luong/' + name, 'w', encoding='utf-8')
    f.write(text)
    f.close()

def preprocess_inp_tar(word):
    word_list = word.split(' ')
    return ' '.join(word_list[1:-1])

def lines_to_text(lines, sep):
    text = ''
    for i in range(len(lines)):
        if i == len(lines) - 1:
            text += str(lines[i])
        else:
            text += str(lines[i]) + sep
    return text

def main():
    #model = int(sys.argv[1])
    #file_name = sys.argv[2]
    """val_inp = text_retrieve(file_name+'.en')
    val_tar = text_retrieve(file_name+'.phone')
    inp_lines, tar_lines, pred_lines = [], [], []
    for i in range(len(val_inp)):
        inp = str(val_inp[i])
        tar = str(val_tar[i])
        try:
            pred = translate(inp, model)
        except:
            continue
        print(i)
        print('Input sentence: ', preprocess_inp_tar(inp))
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
    inp = 'n a t i o n a l i z a t i o n'
    print(inp)
    print(translate(inp, 7))
    print()
    inp = 'k o r z e n i e w s k i'
    print(inp)
    print(translate(inp, 7))
    print()
    inp = 'g r a n d f a t h e r s'
    print(inp)
    print(translate(inp, 7))
    print()

main()
