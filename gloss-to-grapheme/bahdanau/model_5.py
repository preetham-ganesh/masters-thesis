import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, emb_size, rnn_size, vocab_size, rate):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, emb_size)
        self.rnn_fwd = tf.keras.layers.LSTM(rnn_size//2, return_state=True, return_sequences=True)
        self.rnn_bwd = tf.keras.layers.LSTM(rnn_size//2, return_state=True, return_sequences=True, go_backwards=True)
        self.bi_rnn = tf.keras.layers.Bidirectional(self.rnn_fwd, backward_layer=self.rnn_bwd)
        self.rnn_2 = tf.keras.layers.LSTM(rnn_size, return_state=True, return_sequences=True)
        self.dropout = tf.keras.layers.Dropout(rate=rate)

    def call(self, x, training, hidden):
        x = self.embedding(x)
        x, fwd_h, fwd_c, bwd_h, bwd_c = self.bi_rnn(x, initial_state=hidden)
        x = self.dropout(x, training=training)
        h = tf.concat([fwd_h, bwd_h], axis=1)
        del fwd_h, bwd_h
        c = tf.concat([fwd_c, bwd_c], axis=1)
        del fwd_c, bwd_c
        x = self.dropout(x, training=training)
        x, h, c = self.rnn_2(x, initial_state=[h, c])
        x = self.dropout(x, training=training)
        return x, [h, c]

    def initialize_hidden_state(self, batch_size, units):
        fwd_h = tf.zeros((batch_size, int(units//2)))
        bwd_h = tf.zeros((batch_size, int(units//2)))
        fwd_c = tf.zeros((batch_size, int(units//2)))
        bwd_c = tf.zeros((batch_size, int(units//2)))
        return [fwd_h, fwd_c, bwd_h, bwd_c]

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.W3 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, enc_out, h, c):
        h_time = tf.expand_dims(h, 1)
        c_time = tf.expand_dims(c, 1)
        score = self.V(tf.nn.tanh(self.W1(enc_out) + self.W2(h_time) + self.W3(c_time)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * enc_out
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

class Decoder(tf.keras.Model):
    def __init__(self, emb_size, rnn_size, tar_vocab_size, rate):
        super(Decoder, self).__init__()
        self.attention = BahdanauAttention(rnn_size)
        self.embedding = tf.keras.layers.Embedding(tar_vocab_size, emb_size)
        self.rnn_1 = tf.keras.layers.LSTM(rnn_size, return_state=True, return_sequences=True)
        self.rnn_2 = tf.keras.layers.LSTM(rnn_size, return_state=True, return_sequences=True)
        self.rnn_3 = tf.keras.layers.LSTM(rnn_size, return_state=True, return_sequences=True)
        self.dropout = tf.keras.layers.Dropout(rate=rate)
        self.ws = tf.keras.layers.Dense(tar_vocab_size)

    def call(self, x, hidden, enc_out, training):
        context_vector = self.attention(enc_out, hidden[0], hidden[1])
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        x, h, c = self.rnn_1(x)
        x = self.dropout(x, training=training)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        x, h, c = self.rnn_2(x, initial_state=[h, c])
        x = self.dropout(x, training=training)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        x, h, c = self.rnn_3(x, initial_state=[h, c])
        x = self.dropout(x, training=training)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.ws(x)
        return x, [h, c]