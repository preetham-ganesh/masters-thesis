import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, emb_size, rnn_size, vocab_size, rate):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, emb_size)
        self.rnn_1 = tf.keras.layers.LSTM(rnn_size, return_state=True, return_sequences=True)
        self.rnn_2 = tf.keras.layers.LSTM(rnn_size, return_state=True, return_sequences=True)
        self.dropout = tf.keras.layers.Dropout(rate=rate)

    def call(self, x, training, hidden):
        x = self.embedding(x)
        x, h, c = self.rnn_1(x, initial_state=hidden)
        x = self.dropout(x, training=training)
        x, h, c = self.rnn_2(x, initial_state=[h, c])
        x = self.dropout(x, training=training)
        return x, [h, c]

    def initialize_hidden_state(self, batch_size, units):
        h = tf.zeros((batch_size, int(units)))
        c = tf.zeros((batch_size, int(units)))
        return [h, c]

class LuongAttention(tf.keras.Model):
    def __init__(self, rnn_size):
        super(LuongAttention, self).__init__()
        self.wa = tf.keras.layers.Dense(rnn_size)

    def call(self, dec_out, enc_out):
        score = tf.matmul(dec_out, self.wa(enc_out), transpose_b=True)
        alignment = tf.nn.softmax(score, axis=2)
        context = tf.matmul(alignment, enc_out)
        return context

class Decoder(tf.keras.Model):
    def __init__(self, emb_size, rnn_size, tar_vocab_size, rate):
        super(Decoder, self).__init__()
        self.attention = LuongAttention(rnn_size)
        self.embedding = tf.keras.layers.Embedding(tar_vocab_size, emb_size)
        self.rnn_1 = tf.keras.layers.LSTM(rnn_size, return_state=True, return_sequences=True)
        self.rnn_2 = tf.keras.layers.LSTM(rnn_size, return_state=True, return_sequences=True)
        self.wc = tf.keras.layers.Dense(rnn_size, activation='tanh')
        self.dropout = tf.keras.layers.Dropout(rate=rate)
        self.ws = tf.keras.layers.Dense(tar_vocab_size)

    def call(self, x, hidden, enc_out, training):
        x = self.embedding(x)
        x, h, c = self.rnn_1(x, initial_state=hidden)
        x = self.dropout(x, training=training)
        x, h, c = self.rnn_2(x, initial_state=hidden)
        x = self.dropout(x, training=training)
        context = self.attention(x, enc_out)
        x = tf.concat([tf.squeeze(context, 1), tf.squeeze(x, 1)], 1)
        x = self.wc(x)
        x = self.dropout(x, training=training)
        x = self.ws(x)
        return x, [h, c]