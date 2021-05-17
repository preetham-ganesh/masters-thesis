import tensorflow as tf
import pickle
import time
from model_1 import Encoder as encoder_1
from model_1 import Decoder as decoder_1
from model_2 import Encoder as encoder_2
from model_2 import Decoder as decoder_2
from model_3 import Encoder as encoder_3
from model_3 import Decoder as decoder_3
from model_4 import Encoder as encoder_4
from model_4 import Decoder as decoder_4
from model_5 import Encoder as encoder_5
from model_5 import Decoder as decoder_5
from model_6 import Encoder as encoder_6
from model_6 import Decoder as decoder_6
from model_7 import Encoder as encoder_7
from model_7 import Decoder as decoder_7
from model_8 import Encoder as encoder_8
from model_8 import Decoder as decoder_8
import pandas as pd
import re

def create_new_dataset(inp, tar, max_length):
    new_inp, new_tar = [], []
    for i, j in zip(inp, tar):
        if len(i.split(' ')) <= max_length and len(j.split(' ')) <= max_length:
            new_inp.append(i)
            new_tar.append(j)
    return new_inp, new_tar

def tokenize(train, val, test, maxlen):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(train)
    train_tensor = lang_tokenizer.texts_to_sequences(train)
    train_tensor = tf.keras.preprocessing.sequence.pad_sequences(train_tensor, padding='post', maxlen=maxlen)
    val_tensor = lang_tokenizer.texts_to_sequences(val)
    val_tensor = tf.keras.preprocessing.sequence.pad_sequences(val_tensor, padding='post', maxlen=maxlen)
    test_tensor = lang_tokenizer.texts_to_sequences(test)
    test_tensor = tf.keras.preprocessing.sequence.pad_sequences(test_tensor, padding='post', maxlen=maxlen)
    return lang_tokenizer, train_tensor, val_tensor, test_tensor

def text_retrieve(name):
    with open('/home/preetham/Documents/Preetham/masters-thesis/data/grapheme-to-phoneme/cleaned/'+name, 'r', encoding='utf-8') as f:
        text = f.read()
    f.close()
    return text.split('\n')

def open_file(name):
    loc_to = '/home/preetham/Documents/Preetham/masters-thesis/results/grapheme-to-phoneme/bahdanau/'
    with open(loc_to + name + '.pkl', 'rb') as f:
        d = pickle.load(f)
    f.close()
    return d

def save_file(d, name):
    loc_to = '/home/preetham/Documents/Preetham/masters-thesis/results/grapheme-to-phoneme/bahdanau/'
    with open(loc_to + name + '.pkl', 'wb') as f:
        pickle.dump(d, f, protocol=2)
    print(name + ' saved successfully')
    f.close()

def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

@tf.function
def train_step(inp, tar, encoder, decoder, optimizer, tar_word_index, batch_size, hidden):
    loss = 0
    with tf.GradientTape() as tape:
        enc_out, enc_hidden = encoder(inp, True, hidden)
        dec_hidden = enc_hidden
        dec_inp = tf.expand_dims([tar_word_index['<s>']] * batch_size, 1)
        for i in range(1, tar.shape[1]):
            prediction, dec_hidden = decoder(dec_inp, dec_hidden, enc_out, True)
            loss += loss_function(tar[:, i], prediction)
            dec_inp = tf.expand_dims(tar[:, i], 1)
    batch_loss = loss / tar.shape[1]
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    train_loss(batch_loss)

def validation_step(inp, tar, encoder, decoder, tar_word_index, batch_size, hidden):
    loss = 0
    enc_out, enc_hidden = encoder(inp, False, hidden)
    dec_hidden = enc_hidden
    dec_inp = tf.expand_dims([tar_word_index['<s>']] * batch_size, 1)
    for i in range(1, tar.shape[1]):
        prediction, dec_hidden = decoder(dec_inp, dec_hidden, enc_out, False)
        loss += loss_function(tar[:, i], prediction)
        dec_inp = tf.expand_dims(tar[:, i], 1)
    batch_loss = loss / tar.shape[1]
    val_loss(batch_loss)

def choose_encoder_decoder(parameters):
    emb_size = parameters['emb_size']
    inp_vocab_size = parameters['inp_vocab_size']
    tar_vocab_size = parameters['tar_vocab_size']
    rnn_size = parameters['rnn_size']
    rate = parameters['rate']
    if parameters['model'] == 1:
        encoder = encoder_1(emb_size, rnn_size, inp_vocab_size, rate)
        decoder = decoder_1(emb_size, rnn_size, tar_vocab_size, rate)
    elif parameters['model'] == 2:
        encoder = encoder_2(emb_size, rnn_size, inp_vocab_size, rate)
        decoder = decoder_2(emb_size, rnn_size, tar_vocab_size, rate)
    elif parameters['model'] == 3:
        encoder = encoder_3(emb_size, rnn_size, inp_vocab_size, rate)
        decoder = decoder_3(emb_size, rnn_size, tar_vocab_size, rate)
    elif parameters['model'] == 4:
        encoder = encoder_4(emb_size, rnn_size, inp_vocab_size, rate)
        decoder = decoder_4(emb_size, rnn_size, tar_vocab_size, rate)
    elif parameters['model'] == 5:
        encoder = encoder_5(emb_size, rnn_size, inp_vocab_size, rate)
        decoder = decoder_5(emb_size, rnn_size, tar_vocab_size, rate)
    elif parameters['model'] == 6:
        encoder = encoder_6(emb_size, rnn_size, inp_vocab_size, rate)
        decoder = decoder_6(emb_size, rnn_size, tar_vocab_size, rate)
    elif parameters['model'] == 7:
        encoder = encoder_7(emb_size, rnn_size, inp_vocab_size, rate)
        decoder = decoder_7(emb_size, rnn_size, tar_vocab_size, rate)
    elif parameters['model'] == 8:
        encoder = encoder_8(emb_size, rnn_size, inp_vocab_size, rate)
        decoder = decoder_8(emb_size, rnn_size, tar_vocab_size, rate)
    return encoder, decoder

def model_training(train_dataset, val_dataset, parameters):
    global train_loss, val_loss
    loc_to = '/home/preetham/Documents/Preetham/masters-thesis/results/grapheme-to-phoneme/bahdanau/'
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    encoder, decoder = choose_encoder_decoder(parameters)
    optimizer = tf.keras.optimizers.Adam()
    checkpoint_dir = loc_to + 'model_' + str(parameters['model']) + '/training_checkpoints'
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
    manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=3)
    tar_word_index = open_file('model_' + str(parameters['model']) + '/utils/tar-word-index')
    split_df = pd.DataFrame(columns=['epochs', 'train_loss', 'val_loss'])
    best_val_loss = None
    checkpoint_count = 0
    for epoch in range(parameters['epochs']):
        hidden = encoder.initialize_hidden_state(parameters['batch_size'], parameters['rnn_size'])
        train_loss.reset_states()
        val_loss.reset_states()
        epoch_start = time.time()
        for (batch, (inp, tar)) in enumerate(train_dataset.take(parameters['train_steps_per_epoch'])):
            batch_start = time.time()
            train_step(inp, tar, encoder, decoder, optimizer, tar_word_index, parameters['batch_size'], hidden)
            batch_end = time.time()
            if batch % 50 == 0:
                print('Epoch=' + str(epoch+1) + ', Batch=' + str(batch) + ', Training Loss=' +
                      str(round(train_loss.result().numpy(), 3)) + ', Time taken=' +
                      str(round(batch_end - batch_start, 3)) + ' sec')
        for (batch, (inp, tar)) in enumerate(val_dataset.take(parameters['val_steps_per_epoch'])):
            batch_start = time.time()
            validation_step(inp, tar, encoder, decoder, tar_word_index, parameters['batch_size'], hidden)
            batch_end = time.time()
            if batch % 10 == 0:
                print('Epoch=' + str(epoch + 1) + ', Batch=' + str(batch) + ', Validation Loss=' +
                      str(round(val_loss.result().numpy(), 3)) + ', Time taken=' +
                      str(round(batch_end - batch_start, 3)) + ' sec')
        d = {'epochs': int(epoch)+1, 'train_loss': train_loss.result().numpy(), 'val_loss': val_loss.result().numpy()}
        split_df = split_df.append(d, ignore_index=True)
        split_df.to_csv(loc_to + 'model_' + str(parameters['model']) + '/history/split_steps.csv', index=False)
        print()
        print('Epoch=' + str(epoch+1) + ', Training Loss=' + str(round(train_loss.result().numpy(), 3)) +
              ', Validation Loss=' + str(round(val_loss.result().numpy(), 3))+', Time Taken='+
              str(round(time.time()-epoch_start, 3))+ ' sec')
        if best_val_loss is None:
            checkpoint_count = 0
            best_val_loss = round(val_loss.result().numpy(), 3)
            manager.save()
            print('Checkpoint saved')
            print()
        elif best_val_loss > round(val_loss.result().numpy(), 3):
            checkpoint_count = 0
            print('Best Validation Loss changed from ' + str(best_val_loss) + ' to ' +
                  str(round(val_loss.result().numpy(), 3)))
            best_val_loss = round(val_loss.result().numpy(), 3)
            manager.save()
            print('Checkpoint saved')
            print()
        elif checkpoint_count <= 4:
            checkpoint_count += 1
            print('Best Validation Loss did not improve')
            print('Checkpoint not saved')
            print()
        else:
            print('Model did not improve after 4th time. Model stopped from training further.')
            print()
            break

def model_testing(test_dataset, parameters):
    global val_loss
    loc_to = '/home/preetham/Documents/Preetham/masters-thesis/results/grapheme-to-phoneme/bahdanau/'
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_loss.reset_states()
    encoder, decoder = choose_encoder_decoder(parameters)
    checkpoint_dir = loc_to + 'model_' + str(parameters['model']) + '/training_checkpoints'
    checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    tar_word_index = open_file('model_' + str(parameters['model']) + '/utils/tar-word-index')
    hidden = encoder.initialize_hidden_state(parameters['batch_size'], parameters['rnn_size'])
    for (batch, (inp, tar)) in enumerate(test_dataset.take(parameters['test_steps'])):
            validation_step(inp, tar, encoder, decoder, tar_word_index, parameters['batch_size'], hidden)
    print('Test Loss=', round(val_loss.result().numpy(), 3))
    print()

def preprocess_sentence(w):
    w = re.sub(r"[^a-z'\"]+", " ", w)
    return w

def translate(word, model_name):
    loc_to = '/home/preetham/Documents/Preetham/masters-thesis/results/grapheme-to-phoneme/bahdanau/'
    tar_word_index = open_file('model_' + str(model_name) + '/utils/tar-word-index')
    inp_word_index = open_file('model_' + str(model_name) + '/utils/inp-word-index')
    tar_index_word = open_file('model_' + str(model_name) + '/utils/tar-index-word')
    parameters = open_file('model_' + str(model_name) + '/utils/parameters')
    sequence = [[inp_word_index[i] for i in word.split(' ')]]
    sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=parameters['max_length'], padding='post')
    sequence = tf.convert_to_tensor(sequence)
    phoneme = []
    encoder, decoder = choose_encoder_decoder(parameters)
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
