import os
import logging
import tensorflow as tf
from utils import translate, text_retrieve
import re
import unicodedata
import spacy
import math
import operator
from functools import reduce

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

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
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(w)
    rare_words = []
    for i in doc:
        if i.tag_ == 'NNP':
            rare_words.append(i.text)
    new_w = []
    for i in w.split(' '):
        if i in rare_words:
            new_w.append('<'+'#'.join(i)+'>')
        else:
            new_w.append(i)
    w = ' '.join(new_w)
    return w

def preprocess_pred(sentence):
    sentence = sentence.replace('<', '')
    sentence = sentence.replace('#', '')
    sentence = sentence.replace('>', '')
    return sentence

def main():
    inp = ' DESC-N CHIEF INVESTIGATOR X-HIMSELF BE TARGET AND HOUSE CARD COLLAPSE .'.lower()
    gt = 'then the chief investigator himself is targeted and the house of cards collapses . '.lower()
    new_inp = preprocess_sentence(inp)
    out = preprocess_pred(translate(new_inp, 3))
    print()
    print('Processed input: ', inp)
    print('Ground Truth: ', gt)
    print('Translated output: ', out)
    print()

main()
