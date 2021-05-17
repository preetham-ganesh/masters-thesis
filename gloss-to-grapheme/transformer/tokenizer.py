import tensorflow_datasets as tfds
from utils import text_retrieve
import pickle

def create_new_dataset(inp, tar, max_length):
    new_inp, new_tar = [], []
    for i, j in zip(inp, tar):
        if len(i) <= max_length and len(j) <= max_length:
            new_inp.append(i)
            new_tar.append(j)
    return new_inp, new_tar

def save_file(d, name):
    loc_to = '/home/preetham/Documents/Preetham/masters-thesis/data/gloss-to-grapheme/swt-tokenized/'
    with open(loc_to + name + '.pkl', 'wb') as f:
        pickle.dump(d, f, protocol=2)
    print(name + ' saved successfully')
    f.close()

def text_encoder(tokenizer_inp, tokenizer_tar, inp_lines, tar_lines, max_length):
    inp_lines = [[tokenizer_inp.vocab_size] + tokenizer_inp.encode(i) + [tokenizer_inp.vocab_size + 1] for i in
                 inp_lines]
    tar_lines = [[tokenizer_tar.vocab_size] + tokenizer_tar.encode(i) + [tokenizer_tar.vocab_size + 1] for i in
                 tar_lines]
    inp_lines, tar_lines = create_new_dataset(inp_lines, tar_lines, max_length)
    return inp_lines, tar_lines

def tokenizer(train_inp, train_tar, val_inp, val_tar, test_inp, test_tar, vocab_size, max_length):
    tokenizer_inp = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus((i for i in train_inp),
                                                                            target_vocab_size=vocab_size)
    print('Input Tokenizer trained')
    print()
    print('Input Tokenizer Vocab size: ', tokenizer_inp.vocab_size+2)
    print()
    tokenizer_tar = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus((i for i in train_tar),
                                                                            target_vocab_size=vocab_size)
    print('Target Tokenizer trained')
    print()
    print('Input Tokenizer Vocab size: ', tokenizer_tar.vocab_size + 2)
    print()
    tokenizer_inp.save_to_file(
        '/home/preetham/Documents/Preetham/masters-thesis/results/gloss-to-grapheme/tokenizer/gloss-swt')
    print('Input Tokenizer saved')
    print()
    tokenizer_tar.save_to_file(
        '/home/preetham/Documents/Preetham/masters-thesis/results/gloss-to-grapheme/tokenizer/en-swt')
    print('Target Tokenizer saved')
    print()
    train_inp, train_tar = text_encoder(tokenizer_inp, tokenizer_tar, train_inp, train_tar, max_length)
    print('No. of lines after encoding sentences in Training set: ', len(train_inp))
    print()
    save_file(train_inp, 'train.gloss')
    print()
    save_file(train_tar, 'train.en')
    print()
    val_inp, val_tar = text_encoder(tokenizer_inp, tokenizer_tar, val_inp, val_tar, max_length)
    print('No. of lines after encoding sentences in Validation set: ', len(val_inp))
    print()
    save_file(val_inp, 'val.gloss')
    print()
    save_file(val_tar, 'val.en')
    print()
    test_inp, test_tar = text_encoder(tokenizer_inp, tokenizer_tar, test_inp, test_tar, max_length)
    print('No. of lines after encoding sentences in Testing set: ', len(test_inp))
    print()
    save_file(test_inp, 'test.gloss')
    print()
    save_file(test_tar, 'test.en')
    print()

def main():
    print()
    train_inp = text_retrieve('train.gloss')
    val_inp = text_retrieve('val.gloss')
    test_inp = text_retrieve('test.gloss')
    train_tar = text_retrieve('train.en')
    val_tar = text_retrieve('val.en')
    test_tar = text_retrieve('test.en')
    print('No. of original sentences in Training set: ', len(train_inp))
    print('No. of original sentences in Validation set: ', len(val_inp))
    print('No. of original sentences in Test set: ', len(test_inp))
    print()
    vocab_size = 4000
    max_length = 40
    tokenizer(train_inp, train_tar, val_inp, val_tar, test_inp, test_tar, vocab_size, max_length)

main()