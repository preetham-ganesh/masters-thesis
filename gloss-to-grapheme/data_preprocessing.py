import pandas as pd
import re
from random import shuffle
from collections import Counter
import unicodedata
import spacy
import numpy as np

def lines_to_text(lines, sep):
    text = ''
    for i in range(len(lines)):
        if i == len(lines) - 1:
            text += str(lines[i])
        else:
            text += str(lines[i]) + sep
    return text

def unicode_to_ascii(s):
    s = unicodedata.normalize('NFD', s)
    s = s.encode('ASCII', 'ignore')
    return s

def remove_html_markup(s):
    tag = False
    quote = False
    out = ""
    for c in s:
        if c == '<' and not quote:
            tag = True
        elif c == '>' and not quote:
            tag = False
        elif (c == '"' or c == "'") and tag:
            quote = not quote
        elif not tag:
            out = out + c
    return out

def preprocess_sentence(w):
    w = remove_html_markup(w)
    w = unicode_to_ascii(w.lower().strip()).decode('utf-8')
    w = w.replace('”', '"')
    l = ['desc-', 'x-', '`', '[', ']', '-', '/']
    for i in l:
        w = w.replace(i, '')
    w = re.sub("[\(\[].*?[\)\]]", "", w)
    if w == '' or '*' in w or '#' in w or '=' in w or '+' in w or '%' in w:
        return 0
    w = re.sub("[A-Za-z]+", lambda ele: " " + ele[0] + " ", w)
    w = w.strip()
    w = re.sub(r'\s+', ' ', w)
    return w

def file_retrieve(name):
    with open('/home/preetham/Documents/Preetham/masters-thesis/data/gloss-to-grapheme/original/'+name, encoding='utf-8-sig') as f:
        text = f.read()
    f.close()
    return text.split('\n')

def create_dataset(inp_lines, tar_lines):
    new_inp_lines, new_tar_lines = [], []
    for i, j in zip(inp_lines, tar_lines):
        if pd.isnull(i) or pd.isnull(j):
            continue
        else:
            new_i = preprocess_sentence(i)
            new_j = preprocess_sentence(j)
            if new_i == 0 or new_j == 0:
                continue
            else:
                new_j = new_j.replace('rrb', '')
                new_j = new_j.replace('lrb', '')
                new_inp_lines.append(new_i)
                new_tar_lines.append(new_j)
    return new_inp_lines, new_tar_lines

def text_stat(lines):
    text = lines_to_text(lines, ' ')
    letters = Counter(text)
    print('No. of unique characters in text: ', len(letters.keys()))
    words = Counter(text.split(' '))
    print('No. of unique words in text: ', len(words.keys()))
    return words, letters

def find_rare_words(words):
    rare_words = []
    for w in words.keys():
        if words[w] == 1 and not w.isdigit() and w.isalpha():
            rare_words.append(w)
    return rare_words

def oov_handling(inp_lines, tar_lines):
    new_inp_lines, new_tar_lines = [], []
    nlp = spacy.load("en_core_web_sm")
    for i, j in zip(inp_lines, tar_lines):
        rare_words = []
        doc = nlp(i)
        for k in doc:
            if k.tag_ == 'NNP':
                rare_words.append(k.text)
        rare_words = list(set(i.split(' ')) & set(rare_words) & set(j.split(' ')))
        if len(rare_words) > 1:
            new_rare_words = []
            for k in rare_words:
                p = np.random.randint(0, 100, 1)[0] / 100
                if p >= 0.5:
                    new_rare_words.append(k)
                    
            rare_words = new_rare_words
        new_inp, new_tar = [], []
        for l in i.split(' '):
            if l in rare_words:
                new_inp.append('<'+'#'.join(l)+'>')
            else:
                new_inp.append(l)
        for l in j.split(' '):
            if l in rare_words:
                new_tar.append('<'+'#'.join(l)+'>')
            else:
                new_tar.append(l)
        new_inp = ' '.join(new_inp)
        new_tar = ' '.join(new_tar)
        new_inp_lines.append(new_inp)
        new_tar_lines.append(new_tar)
    return new_inp_lines, new_tar_lines

def dataset_save(lines, name):
    text = lines_to_text(lines, '\n')
    f = open('/home/preetham/Documents/Preetham/masters-thesis/data/gloss-to-grapheme/' + name, 'w', encoding='utf-8')
    f.write(text)
    f.close()
    print(name + ' saved successfully')
    print()

def shuffle_lines(inp_lines, tar_lines):
    c = list(zip(inp_lines, tar_lines))
    shuffle(c)
    inp_lines, tar_lines = zip(*c)
    return inp_lines, tar_lines

def drop_duplicates(inp_lines, tar_lines):
    d = {'inp': inp_lines, 'tar': tar_lines}
    df = pd.DataFrame(d)
    df = df.drop_duplicates()
    if len(df.inp.unique()) < len(df.tar.unique()):
        df = df.drop_duplicates(subset='inp', keep='first')
    else:
        df = df.drop_duplicates(subset='tar', keep='first')
    return list(df['inp']), list(df['tar'])

def inp_tar_lines_stats(inp_lines, tar_lines):
    print('No. of English lines in dataset: ', len(inp_lines))
    print('No. of Gloss lines in dataset: ', len(tar_lines))
    print()
    print('English lines stats')
    print()
    _, _ = text_stat(inp_lines)
    print()
    print('Gloss lines stats')
    print()
    _, _ = text_stat(tar_lines)

def main():
    print()
    data_en = file_retrieve('data.en')
    data_gloss = file_retrieve('data.gloss')
    print('Text files retrieved')
    print()
    inp_tar_lines_stats(data_en, data_gloss)
    print()
    data_en, data_gloss = create_dataset(data_en, data_gloss)
    print('Datasets cleaned')
    print()
    inp_tar_lines_stats(data_en, data_gloss)
    print()
    data_en, data_gloss = drop_duplicates(data_en, data_gloss)
    print('Duplicates dropped')
    print()
    inp_tar_lines_stats(data_en, data_gloss)
    print()
    data_en, data_gloss = shuffle_lines(data_en, data_gloss)
    data_en, data_gloss = oov_handling(data_en, data_gloss)
    print('POS Tagging done')
    print()
    inp_tar_lines_stats(data_en, data_gloss)
    print()
    test_en, test_gloss = data_en[:2000], data_gloss[:2000]
    val_en, val_gloss = data_en[2000:4000], data_gloss[2000:4000]
    train_en, train_gloss = data_en[4000:], data_gloss[4000:]
    del data_en, data_gloss
    print('Training set details')
    print()
    inp_tar_lines_stats(train_en, train_gloss)
    print()
    print('Validation set details')
    print()
    inp_tar_lines_stats(val_en, val_gloss)
    print()
    print('Testing set details')
    print()
    inp_tar_lines_stats(test_en, test_gloss)
    print()
    dataset_save(train_en, 'cleaned/train.en')
    dataset_save(train_gloss, 'cleaned/train.gloss')
    dataset_save(val_en, 'cleaned/val.en')
    dataset_save(val_gloss, 'cleaned/val.gloss')
    dataset_save(test_en, 'cleaned/test.en')
    dataset_save(test_gloss, 'cleaned/test.gloss')

main()
