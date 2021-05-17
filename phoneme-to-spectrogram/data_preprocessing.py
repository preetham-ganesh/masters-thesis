import os
import librosa
import pickle
import numpy as np
from import_file import import_file
import re
import tensorflow as tf
import logging
import time
import matplotlib.pyplot as plt

model_7 = import_file('/home/preetham/Documents/Preetham/masters-thesis/codes/grapheme-to-phoneme/luong/model_7.py')
from model_7 import Encoder, Decoder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

mel_filter = librosa.filters.mel(sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=8000)

def open_file(name):
    loc_to = '/home/preetham/Documents/Preetham/masters-thesis/'
    with open(loc_to + name, 'rb') as f:
        d = pickle.load(f)
    f.close()
    return d

def save_file(d, name):
    loc_to = '/home/preetham/Documents/Preetham/masters-thesis/'
    with open(loc_to + name + '.pkl', 'wb') as f:
        pickle.dump(d, f, protocol=2)
    f.close()

def log_magnitude(x):
    x = np.maximum(x, 1e-7)
    x = np.log(x)
    return x

def process_audio(speech):
    D = librosa.core.stft(speech, n_fft=1024, hop_length=256, win_length=1024, window='hann', center=True,
                          pad_mode='reflect')
    S = np.abs(D)
    mel = np.dot(mel_filter, S)
    mel = log_magnitude(mel)
    return mel

def preprocess_sentence(w):
    w = re.sub(r"[^a-z-]+", " ", w)
    w = w.strip()
    w = re.sub(r'\s+', ' ', w)
    return w

def preprocess_inp_tar(word):
    word_list = word.split(' ')
    return ' '.join(word_list[1:-1])

def translate(word, model_name):
    loc_to = '/home/preetham/Documents/Preetham/masters-thesis/results/grapheme-to-phoneme/luong/'
    tar_word_index = open_file('results/grapheme-to-phoneme/luong/model_' + str(model_name) + '/utils/tar-word-index.pkl')
    inp_word_index = open_file('results/grapheme-to-phoneme/luong/model_' + str(model_name) + '/utils/inp-word-index.pkl')
    tar_index_word = open_file('results/grapheme-to-phoneme/luong/model_' + str(model_name) + '/utils/tar-index-word.pkl')
    parameters = open_file('results/grapheme-to-phoneme/luong/model_' + str(model_name) + '/utils/parameters.pkl')
    emb_size = parameters['emb_size']
    inp_vocab_size = parameters['inp_vocab_size']
    tar_vocab_size = parameters['tar_vocab_size']
    rnn_size = parameters['rnn_size']
    rate = parameters['rate']
    encoder = Encoder(emb_size, rnn_size, inp_vocab_size, rate)
    decoder = Decoder(emb_size, rnn_size, tar_vocab_size, rate)
    sequence = [[inp_word_index[i] for i in word.split(' ')]]
    sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=parameters['max_length'], padding='post')
    sequence = tf.convert_to_tensor(sequence)
    phoneme = []
    checkpoint_dir = loc_to + 'model_' + str(parameters['model']) + '/training_checkpoints'
    checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    hidden = encoder.initialize_hidden_state(1, parameters['rnn_size'])
    enc_out, enc_hidden = encoder(sequence, False, hidden)
    dec_hidden = enc_hidden
    dec_inp = tf.expand_dims([tar_word_index['<s>']], 0)
    for i in range(1, parameters['max_length']):
        prediction, dec_hidden = decoder(dec_inp, dec_hidden, enc_out, False)
        predicted_id = tf.argmax(prediction[0]).numpy()
        if tar_index_word[predicted_id] != '</s>':
            phoneme.append(tar_index_word[predicted_id])
        else:
            break
        dec_inp = tf.expand_dims([predicted_id], 0)
    phoneme = ' '.join(phoneme)
    return phoneme

def convert_to_phoneme(text):
    text = text.lower()
    text = preprocess_sentence(text)
    l = text.split(' ')
    new_l = []
    for i in l:
        new_l.append(translate(' '.join(list(i)), 7))
    new_l = '  '.join(new_l)
    return new_l

def lines_to_text(lines, sep):
    text = ''
    for i in range(len(lines)):
        if i == len(lines) - 1:
            text += str(lines[i])
        else:
            text += str(lines[i]) + sep
    return text

def dataset_save(lines, name):
    text = lines_to_text(lines, '\n')
    f = open('/home/preetham/Documents/Preetham/masters-thesis/data/phoneme-to-spectrogram/cleaned/' + name, 'w', encoding='utf-8')
    f.write(text)
    f.close()
    print(name + ' saved successfully')

def file_length_check(files):
    length = {}
    loc = '/home/preetham/Documents/Preetham/masters-thesis/data/phoneme-to-spectrogram/'
    for i in files:
        if os.path.exists(loc + 'cleaned/' + i):
            d = open_file('data/phoneme-to-spectrogram/cleaned/' + i)
            l = d['phoneme'].split('  ')
            new_l = [i.split(' ') for i in l]
            new_l = sum(new_l, [])
            if len(new_l) in list(length.keys()):
                length[len(new_l)] += 1
            else:
                length[len(new_l)] = 1
    x = list(length.keys())
    x.sort()
    y = [length[i] for i in x]
    plt.plot(x, y)
    plt.xlabel('Length of phonemes')
    plt.ylabel('Number of sentences')
    plt.show()
    new_files = []
    for i in files:
        if os.path.exists(loc + 'cleaned/' + i):
            d = open_file('data/phoneme-to-spectrogram/cleaned/' + i)
            l = d['phoneme'].split('  ')
            new_l = [i.split(' ') for i in l]
            new_l = sum(new_l, [])
            if len(new_l) <= 80:
                new_files.append(i)
    print('No. of lines in new dataset: ', len(new_files))
    print()
    dataset_save(new_files, 'files_list.txt')
    print()

def file_convert(files):
    c = 0
    length = {}
    loc = '/home/preetham/Documents/Preetham/masters-thesis/data/phoneme-to-spectrogram/'
    for i in files:
        if os.path.exists(loc + 'cleaned/' + i):
            continue
        try:
            start_time = time.time()
            d = open_file('data/phoneme-to-spectrogram/original/' + i)
            mel = process_audio(d['speech'])
            phoneme = convert_to_phoneme(d['text_normalized'])
            id = d['id']
            d = {'id': id, 'mel_spectrogram': mel, 'phoneme': phoneme}
            print('ID: ', id)
            print('Mel_Spectrogram: ', mel.shape)
            print('Phoneme: ', phoneme)
            save_file(d, 'data/phoneme-to-spectrogram/cleaned/' + id)
            print('Time Taken: ', round(time.time() - start_time, 3))
            print()
            c += 1
        except:
            continue
        if c % 50 == 0:
            print(str(c) + ' files saved successfully')
            print()

def main():
    loc = '/home/preetham/Documents/Preetham/masters-thesis/data/phoneme-to-spectrogram/'
    files = os.listdir(loc+'original')
    print()
    print('No. of lines in original dataset: ', len(files))
    print()
    #file_convert(files)
    file_length_check(files)

main()