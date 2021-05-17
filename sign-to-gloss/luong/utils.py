import pickle
import numpy as np
import tensorflow as tf
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

def open_file(name):
    loc_to = '/home/preetham/Documents/Preetham/masters-thesis/'
    with open(loc_to + name + '.pkl', 'rb') as f:
        d = pickle.load(f)
    f.close()
    return d

def save_file(d, name):
    loc_to = '/home/preetham/Documents/Preetham/masters-thesis/'
    with open(loc_to + name + '.pkl', 'wb') as f:
        pickle.dump(d, f, protocol=2)
    print(name + ' saved successfully')
    f.close()

def create_batch(dataset_phrase, dataset_info, n_classes):
    loc = '/home/preetham/Documents/Preetham/masters-thesis/data/sign-to-gloss/cleaned/keypoints/'
    inp_f, tar_f = [], []
    maxi = 0
    for i in dataset_phrase:
        inp_t, tar_t = np.ones((1, 104)), [n_classes]
        for j in i:
            try:
                t = list(str(j.numpy()))
            except:
                t = list(str(j))
            t_ = ['0' for _ in range(5-len(t))]
            t = ''.join(t_ + t)
            x = np.load(loc+t+'.npy')
            x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
            inp_t = np.concatenate((inp_t, x), axis=0)
            try:
                tar_t.append(dataset_info[j.numpy()][0])
            except:
                tar_t.append(dataset_info[j][0])
        inp_t = np.concatenate((inp_t, np.full((1, 104), 2)), axis=0)
        tar_t.append(n_classes + 1)
        if maxi < inp_t.shape[0]:
            maxi = inp_t.shape[0]
        inp_f.append(inp_t)
        tar_f.append(tar_t)
    tar_f = np.array(tar_f)
    new_inp_f = []
    for i, j in zip(inp_f, tar_f):
        inp_t = np.concatenate((i, np.zeros((maxi - len(i), 104))))
        new_inp_f.append(inp_t)
    new_inp_f = np.array(new_inp_f)
    return new_inp_f, tar_f

def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

@tf.function
def train_step(inp, tar, encoder, decoder, optimizer, tar_start, batch_size, hidden):
    loss = 0
    with tf.GradientTape() as tape:
        enc_out, enc_hidden = encoder(inp, True, hidden)
        dec_hidden = enc_hidden
        dec_inp = tf.expand_dims([tar_start] * batch_size, 1)
        for i in range(1, tar.shape[1]):
            prediction, dec_hidden = decoder(dec_inp, dec_hidden, enc_out, True)
            loss += loss_function(tar[:, i], prediction)
            train_acc_1.update_state(tar[:, i], prediction)
            train_acc_5.update_state(tar[:, i], prediction)
            train_acc_10.update_state(tar[:, i], prediction)
            dec_inp = tf.expand_dims(tar[:, i], 1)
    batch_loss = loss / tar.shape[1]
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    train_loss(batch_loss)

def validation_step(inp, tar, encoder, decoder, tar_start, batch_size, hidden):
    loss = 0
    enc_out, enc_hidden = encoder(inp, False, hidden)
    dec_hidden = enc_hidden
    dec_inp = tf.expand_dims([tar_start] * batch_size, 1)
    for i in range(1, tar.shape[1]):
        prediction, dec_hidden = decoder(dec_inp, dec_hidden, enc_out, False)
        loss += loss_function(tar[:, i], prediction)
        val_acc_1.update_state(tar[:, i], prediction)
        val_acc_5.update_state(tar[:, i], prediction)
        val_acc_10.update_state(tar[:, i], prediction)
        dec_inp = tf.expand_dims(tar[:, i], 1)
    batch_loss = loss / tar.shape[1]
    val_loss(batch_loss)

def choose_encoder_decoder(parameters):
    emb_size = parameters['emb_size']
    tar_vocab_size = parameters['tar_vocab_size']
    rnn_size = parameters['rnn_size']
    rate = parameters['rate']
    if parameters['model'] == 1:
        encoder = encoder_1(rnn_size, rate)
        decoder = decoder_1(emb_size, rnn_size, tar_vocab_size, rate)
    elif parameters['model'] == 2:
        encoder = encoder_2(rnn_size, rate)
        decoder = decoder_2(emb_size, rnn_size, tar_vocab_size, rate)
    elif parameters['model'] == 3:
        encoder = encoder_3(rnn_size, rate)
        decoder = decoder_3(emb_size, rnn_size, tar_vocab_size, rate)
    elif parameters['model'] == 4:
        encoder = encoder_4(rnn_size, rate)
        decoder = decoder_4(emb_size, rnn_size, tar_vocab_size, rate)
    elif parameters['model'] == 5:
        encoder = encoder_5(rnn_size, rate)
        decoder = decoder_5(emb_size, rnn_size, tar_vocab_size, rate)
    elif parameters['model'] == 6:
        encoder = encoder_6(rnn_size, rate)
        decoder = decoder_6(emb_size, rnn_size, tar_vocab_size, rate)
    elif parameters['model'] == 7:
        encoder = encoder_7(rnn_size, rate)
        decoder = decoder_7(emb_size, rnn_size, tar_vocab_size, rate)
    elif parameters['model'] == 8:
        encoder = encoder_8(rnn_size, rate)
        decoder = decoder_8(emb_size, rnn_size, tar_vocab_size, rate)
    return encoder, decoder

def model_training(train_dataset, val_dataset, dataset_info, parameters):
    global train_loss, val_loss, train_acc_1, train_acc_5, train_acc_10, val_acc_1, val_acc_5, val_acc_10
    n_classes = parameters['tar_vocab_size'] - 2
    loc_to = '/home/preetham/Documents/Preetham/masters-thesis/results/sign-to-gloss/wlasl-' + str(n_classes) + '/luong/'
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc_1 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name='train_acc_1')
    train_acc_5 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='train_acc_5')
    train_acc_10 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name='train_acc_10')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_acc_1 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name='val_acc_1')
    val_acc_5 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='val_acc_5')
    val_acc_10 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name='val_acc_10')
    encoder, decoder = choose_encoder_decoder(parameters)
    optimizer = tf.keras.optimizers.Adam()
    checkpoint_dir = loc_to + 'model_' + str(parameters['model']) + '/training_checkpoints'
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
    manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=3)
    split_df = pd.DataFrame(columns=['epochs', 'train_loss', 'val_loss'])
    best_val_loss = None
    checkpoint_count = 0
    for epoch in range(parameters['epochs']):
        hidden = encoder.initialize_hidden_state(parameters['batch_size'], parameters['rnn_size'])
        train_loss.reset_states()
        train_acc_1.reset_states()
        train_acc_5.reset_states()
        train_acc_10.reset_states()
        val_loss.reset_states()
        val_acc_1.reset_states()
        val_acc_5.reset_states()
        val_acc_10.reset_states()
        epoch_start = time.time()
        for (batch, x) in enumerate(train_dataset.take(parameters['train_steps_per_epoch'])):
            batch_start = time.time()
            inp, tar = create_batch(x, dataset_info, parameters['tar_vocab_size']-2)
            train_step(inp, tar, encoder, decoder, optimizer, parameters['tar_vocab_size']-2, parameters['batch_size'],
                       hidden)
            batch_end = time.time()
            if batch % 100 == 0:
                print('Epoch=' + str(epoch+1) + ', Batch=' + str(batch) + ', Training Loss=' +
                      str(round(train_loss.result().numpy(), 3)) + ', Training Accuracy 1=' +
                      str(round(train_acc_1.result().numpy(), 3)) + ', Training Accuracy 5=' +
                      str(round(train_acc_5.result().numpy(), 3)) + ', Training Accuracy 10=' +
                      str(round(train_acc_10.result().numpy(), 3)) + ', Time taken=' +
                      str(round(batch_end - batch_start, 3)) + ' sec')
        for (batch, x) in enumerate(val_dataset.take(parameters['val_steps_per_epoch'])):
            batch_start = time.time()
            inp, tar = create_batch(x, dataset_info, parameters['tar_vocab_size'] - 2)
            validation_step(inp, tar, encoder, decoder, parameters['tar_vocab_size']-2, parameters['batch_size'], hidden)
            batch_end = time.time()
            if batch % 10 == 0:
                print('Epoch=' + str(epoch + 1) + ', Batch=' + str(batch) + ', Validation Loss=' +
                      str(round(val_loss.result().numpy(), 3)) + ', Validation Accuracy 1=' +
                      str(round(val_acc_1.result().numpy(), 3)) + ', Validation Accuracy 5=' +
                      str(round(val_acc_5.result().numpy(), 3)) + ', Validation Accuracy 10=' +
                      str(round(val_acc_10.result().numpy(), 3)) + ', Time taken=' +
                      str(round(batch_end - batch_start, 3)) + ' sec')
        d = {'epochs': int(epoch)+1, 'train_loss': train_loss.result().numpy(), 'val_loss': val_loss.result().numpy(),
             'train_acc_1': train_acc_1.result().numpy(), 'val_acc_1': val_acc_1.result().numpy(),
             'train_acc_5': train_acc_1.result().numpy(), 'val_acc_5': val_acc_1.result().numpy(),
             'train_acc_5': train_acc_1.result().numpy(), 'val_acc_5': val_acc_1.result().numpy()}
        split_df = split_df.append(d, ignore_index=True)
        split_df.to_csv(loc_to + 'model_' + str(parameters['model']) + '/history/split_steps.csv', index=False)
        print()
        print('Epoch=' + str(epoch+1)+', Time Taken='+str(round(time.time()-epoch_start, 3))+ ' sec')
        print('Training Loss=' + str(round(train_loss.result().numpy(), 3)) +
              ', Validation Loss=' + str(round(val_loss.result().numpy(), 3)))
        print('Training Accuracy 1=' + str(round(train_acc_1.result().numpy(), 3)) +
              ', Validation Accuracy 1=' + str(round(val_acc_1.result().numpy(), 3)))
        print('Training Accuracy 5=' + str(round(train_acc_5.result().numpy(), 3)) +
              ', Validation Accuracy 5=' + str(round(val_acc_5.result().numpy(), 3)))
        print('Training Accuracy 10=' + str(round(train_acc_10.result().numpy(), 3)) +
              ', Validation Accuracy 10=' + str(round(val_acc_10.result().numpy(), 3)))
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

def model_testing(test_dataset, dataset_info, parameters):
    global val_loss, val_acc_1, val_acc_5, val_acc_10
    n_classes = parameters['tar_vocab_size'] - 2
    loc_to = '/home/preetham/Documents/Preetham/masters-thesis/results/sign-to-gloss/wlasl-' + str(n_classes) + '/luong/'
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_acc_1 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name='val_acc_1')
    val_acc_5 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='val_acc_5')
    val_acc_10 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name='val_acc_10')
    val_loss.reset_states()
    val_acc_1.reset_states()
    val_acc_5.reset_states()
    val_acc_10.reset_states()
    encoder, decoder = choose_encoder_decoder(parameters)
    checkpoint_dir = loc_to + 'model_' + str(parameters['model']) + '/training_checkpoints'
    checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    hidden = encoder.initialize_hidden_state(parameters['batch_size'], parameters['rnn_size'])
    for (batch, x) in enumerate(test_dataset.take(parameters['test_steps'])):
        inp, tar = create_batch(x, dataset_info, parameters['tar_vocab_size'] - 2)
        validation_step(inp, tar, encoder, decoder, parameters['tar_vocab_size'] - 2, parameters['batch_size'], hidden)
    print('Test Loss =', round(val_loss.result().numpy(), 3))
    print('Test Accuracy 1 =', round(val_acc_1.result().numpy()*100, 2))
    print('Test Accuracy 5 =', round(val_acc_5.result().numpy()*100, 2))
    print('Test Accuracy 10 =', round(val_acc_10.result().numpy()*100, 2))
    print()

def translate(keypoints, model_name, n_classes):
    loc_to = '/home/preetham/Documents/Preetham/masters-thesis/results/sign-to-gloss/wlasl-' + str(
        n_classes) + '/luong/'
    parameters = open_file('results/sign-to-gloss/wlasl-' + str(
        n_classes) + '/luong/model_' + str(model_name) + '/utils/parameters')
    encoder, decoder = choose_encoder_decoder(parameters)
    sequence = tf.convert_to_tensor(keypoints)
    checkpoint_dir = loc_to + 'model_' + str(parameters['model']) + '/training_checkpoints'
    checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    hidden = encoder.initialize_hidden_state(1, parameters['rnn_size'])
    enc_out, enc_hidden = encoder(sequence, False, hidden)
    dec_hidden = enc_hidden
    dec_inp = tf.expand_dims([parameters['tar_vocab_size']-2], 0)
    gloss = []
    for i in range(1, sequence.shape[1]):
        prediction, dec_hidden = decoder(dec_inp, dec_hidden, enc_out, False)
        predicted_id = tf.argmax(prediction[0]).numpy()
        if predicted_id != parameters['tar_vocab_size']-1:
            gloss.append(predicted_id)
        else:
            break
        dec_inp = tf.expand_dims([predicted_id], 0)
    return gloss