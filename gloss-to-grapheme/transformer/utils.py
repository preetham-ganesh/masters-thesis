import tensorflow as tf
from model import Transformer
import time
import pickle
import pandas as pd
import tensorflow_datasets as tfds

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def create_new_dataset(inp, tar, max_length):
    new_inp, new_tar = [], []
    for i, j in zip(inp, tar):
        if len(i.split(' ')) <= max_length and len(j.split(' ')) <= max_length:
            new_inp.append(i)
            new_tar.append(j)
    return new_inp, new_tar

def text_retrieve(name):
    with open('/home/preetham/Documents/Preetham/masters-thesis/data/gloss-to-grapheme/cleaned/'+name, 'r',
              encoding='utf-8') as f:
        text = f.read()
    f.close()
    return text.split('\n')

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

def tokenize(train_tensor, val_tensor, test_tensor):
    train_tensor = tf.keras.preprocessing.sequence.pad_sequences(train_tensor, padding='post')
    val_tensor = tf.keras.preprocessing.sequence.pad_sequences(val_tensor, padding='post')
    test_tensor = tf.keras.preprocessing.sequence.pad_sequences(test_tensor, padding='post')
    train_tensor = tf.cast(train_tensor, dtype=tf.int32)
    val_tensor = tf.cast(val_tensor, dtype=tf.int32)
    test_tensor = tf.cast(test_tensor, dtype=tf.int32)
    return train_tensor, val_tensor, test_tensor

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
]
@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    with tf.GradientTape() as tape:
        predictions = transformer(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
        loss = loss_function(tar_real, predictions)
    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    train_loss(loss)
    train_accuracy(tar_real, predictions)

def val_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    predictions = transformer(inp, tar_inp, False, enc_padding_mask, combined_mask, dec_padding_mask)
    loss = loss_function(tar_real, predictions)
    val_loss(loss)
    val_accuracy(tar_real, predictions)

def model_training(train_dataset, val_dataset, parameters):
    loc_to = '/home/preetham/Documents/Preetham/masters-thesis/results/gloss-to-grapheme/transformer/'
    global train_loss, train_accuracy, val_loss, val_accuracy, loss_object, transformer, optimizer
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    learning_rate = CustomSchedule(parameters['d_model'], 4000)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    checkpoint_dir = loc_to + 'model_'+str(parameters['model'])+'/training_checkpoints'
    if parameters['n_layers'] <= 6:
        n_layers = parameters['n_layers']
    else:
        n_layers = parameters['n_layers'] - 6
    transformer = Transformer(n_layers, parameters['d_model'], parameters['n_heads'], parameters['dff'],
                              parameters['inp_vocab_size'], parameters['tar_vocab_size'],
                              pe_input=parameters['inp_vocab_size'], pe_target=parameters['tar_vocab_size'],
                              rate=parameters['dropout'])
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, transformer=transformer)
    manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=3)
    split_df = pd.DataFrame(columns=['epochs', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])
    best_val_loss, best_val_acc = None, None
    checkpoint_count = 0
    for epoch in range(parameters['epochs']):
        epoch_start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()
        for (batch, (inp, tar)) in enumerate(train_dataset.take(parameters['train_steps_per_epoch'])):
            batch_start = time.time()
            train_step(inp, tar)
            batch_end = time.time()
            if batch % 50 == 0:
                print('Epoch=' + str(epoch+1) + ', Batch=' + str(batch) + ', Training Loss=' +
                      str(round(train_loss.result().numpy(), 3)) + ', Training Accuracy=' +
                      str(round(train_accuracy.result().numpy(), 3)) + ', Time taken=' +
                      str(round(batch_end - batch_start, 3)) + ' sec')
        for (batch, (inp, tar)) in enumerate(val_dataset.take(parameters['val_steps_per_epoch'])):
            batch_start = time.time()
            val_step(inp, tar)
            batch_end = time.time()
            if batch % 5 == 0:
                print('Epoch=' + str(epoch + 1) + ', Batch=' + str(batch) + ', Validation Loss=' +
                      str(round(val_loss.result().numpy(), 3)) + ', Validation Accuracy=' +
                      str(round(val_accuracy.result().numpy(), 3)) + ', Time taken=' +
                      str(round(batch_end - batch_start, 3)) + ' sec')
        print()
        print('Epoch=' + str(epoch+1) + ', Training Loss=' + str(round(train_loss.result().numpy(), 3)) +
              ', Validation Loss=' + str(round(val_loss.result().numpy(), 3)) + ', Training Accuracy=' +
              str(round(train_accuracy.result().numpy(), 3)) + ', Validation Accuracy=' +
              str(round(val_accuracy.result().numpy(), 3)) + ', Time taken=' +
              str(round(time.time() - epoch_start, 3)) + ' sec')
        d = {'epochs': int(epoch + 1), 'train_loss': train_loss.result().numpy(),
             'train_acc': train_accuracy.result().numpy(), 'val_loss': val_loss.result().numpy(),
             'val_acc': val_accuracy.result().numpy()}
        split_df = split_df.append(d, ignore_index=True)
        split_df.to_csv(loc_to + 'model_' + str(parameters['model']) + '/history/split_steps.csv', index=False)
        if best_val_loss is None and best_val_acc is None:
            checkpoint_count = 0
            best_val_acc = round(val_accuracy.result().numpy(), 3)
            best_val_loss = round(val_loss.result().numpy(), 3)
            manager.save()
            print('Checkpoint saved')
            print()
        elif round(best_val_loss - round(val_loss.result().numpy(), 3), 3) >= 0.001:
            checkpoint_count = 0
            print('Best Validation Loss changed from ' + str(best_val_loss) + ' to ' +
                  str(round(val_loss.result().numpy(), 3)))
            print('Best Validation Accuracy changed from ' + str(best_val_acc) + ' to ' +
                  str(round(val_accuracy.result().numpy(), 3)))
            best_val_acc = round(val_accuracy.result().numpy(), 3)
            best_val_loss = round(val_loss.result().numpy(), 3)
            manager.save()
            print('Checkpoint saved')
            print()
        elif checkpoint_count <= 4:
            checkpoint_count += 1
            print('Best Validation Loss and Validation Accuracy did not improve')
            print('Checkpoint not saved')
            print()
        else:
            print('Model did not improve after 4th time. Model stopped from training further.')
            print()
            break

def model_testing(test_dataset, parameters):
    loc_to = '/home/preetham/Documents/Preetham/masters-thesis/results/gloss-to-grapheme/transformer/'
    global val_loss, val_accuracy, loss_object, transformer
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
    val_loss.reset_states()
    val_accuracy.reset_states()
    checkpoint_dir = loc_to + 'model_'+str(parameters['model'])+'/training_checkpoints'
    if parameters['n_layers'] <= 6:
        n_layers = parameters['n_layers']
    else:
        n_layers = parameters['n_layers'] - 6
    transformer = Transformer(n_layers, parameters['d_model'], parameters['n_heads'], parameters['dff'],
                              parameters['inp_vocab_size'], parameters['tar_vocab_size'],
                              pe_input=parameters['inp_vocab_size'], pe_target=parameters['tar_vocab_size'],
                              rate=parameters['dropout'])
    checkpoint = tf.train.Checkpoint(transformer=transformer)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    for (batch, (inp, tar)) in enumerate(test_dataset.take(parameters['test_steps'])):
        val_step(inp, tar)
    print('Test Loss=', round(val_loss.result().numpy(), 3))
    print('Test Accuracy=', round(val_accuracy.result().numpy(), 3))
    print()

def translate(gloss, model):
    loc_to = '/home/preetham/Documents/Preetham/masters-thesis/results/gloss-to-grapheme/'
    parameters = open_file('results/gloss-to-grapheme/transformer/model_'+str(model)+'/utils/parameters')
    inp_lang = tfds.deprecated.text.SubwordTextEncoder.load_from_file(loc_to + 'tokenizer/gloss-swt')
    tar_lang = tfds.deprecated.text.SubwordTextEncoder.load_from_file(loc_to + 'tokenizer/en-swt')
    sequence = inp_lang.encode(gloss)
    sequence = [inp_lang.vocab_size] + sequence + [inp_lang.vocab_size + 1]
    sequence = tf.expand_dims(sequence, 0)
    global transformer
    if parameters['n_layers'] <= 6:
        n_layers = parameters['n_layers']
    else:
        n_layers = parameters['n_layers'] - 6
    transformer = Transformer(n_layers, parameters['d_model'], parameters['n_heads'], parameters['dff'],
                              parameters['inp_vocab_size'], parameters['tar_vocab_size'],
                              pe_input=parameters['inp_vocab_size'], pe_target=parameters['tar_vocab_size'],
                              rate=parameters['dropout'])
    checkpoint = tf.train.Checkpoint(transformer=transformer)
    checkpoint_dir = loc_to + 'transformer/model_' + str(parameters['model']) + '/training_checkpoints'
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    dec_inp = tf.expand_dims([tar_lang.vocab_size], 0)
    for i in range(1, 100):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(sequence, dec_inp)
        predictions = transformer(sequence, dec_inp, False, enc_padding_mask, combined_mask, dec_padding_mask)
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        if tar_lang.vocab_size + 1 == predicted_id:
            dec_inp = tf.concat([dec_inp, predicted_id], axis=-1)
            break
        dec_inp = tf.concat([dec_inp, predicted_id], axis=-1)
    dec_inp = tf.squeeze(dec_inp, axis=0)
    sentence = tar_lang.decode([i for i in dec_inp.numpy()[1:-1]])
    return sentence
