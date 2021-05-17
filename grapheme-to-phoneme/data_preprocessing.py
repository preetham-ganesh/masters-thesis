from random import shuffle
import re
import pandas as pd

def lines_to_text(lines, sep):
    text = ''
    for i in range(len(lines)):
        if i == len(lines) - 1:
            text += str(lines[i])
        else:
            text += str(lines[i]) + sep
    return text

def file_retrieve(name, encoding):
    with open('/home/preetham/Documents/Preetham/masters-thesis/data/grapheme-to-phoneme/original/'+name, encoding=encoding) as f:
        text = f.read()
    f.close()
    return text.split('\n')

def preprocess_sentence(w):
    w = re.sub(r"[^a-z-]+", " ", w)
    w = w.strip()
    w = re.sub(r'\s+', ' ', w)
    return w

def create_dataset(data):
    inp_lines, tar_lines = [], []
    c = 0
    for i in data:
        i = i.lower()
        if i != '':
            sep = i.split('  ')
            inp = preprocess_sentence(sep[0])
            inp = ' '.join(list(inp.replace(' ', '')))
            inp = '<s> ' + inp + ' </s>'
            tar = preprocess_sentence(' '.join(sep[1:]))
            tar = '<s> ' + tar + ' </s>'
            inp_lines.append(inp)
            tar_lines.append(tar)
            c += 1
    return inp_lines, tar_lines

def dataset_save(lines, name):
    text = lines_to_text(lines, '\n')
    f = open('/home/preetham/Documents/Preetham/masters-thesis/data/grapheme-to-phoneme/cleaned/' + name, 'w', encoding='utf-8')
    f.write(text)
    f.close()
    print(name + ' saved successfully')

def drop_duplicates(inp_lines, tar_lines):
    d = {'inp': inp_lines, 'tar': tar_lines}
    df = pd.DataFrame(d)
    df = df.drop_duplicates()
    return list(df['inp']), list(df['tar'])

def dataset_convert(inp_lines, tar_lines):
    c = list(zip(inp_lines, tar_lines))
    shuffle(c)
    inp_lines, tar_lines = zip(*c)
    test_inp, test_tar = inp_lines[:2000], tar_lines[:2000]
    val_inp, val_tar = inp_lines[2000:4000], tar_lines[2000:4000]
    inp_lines, tar_lines = inp_lines[4000:], tar_lines[4000:]
    print('No. of lines in the training dataset: ', len(inp_lines))
    print('No. of lines in the validation dataset: ', len(val_inp))
    print('No. of lines in the testing dataset: ', len(test_inp))
    print()
    #dataset_save(inp_lines, 'train.en')
    #dataset_save(tar_lines, 'train.phone')
    #dataset_save(val_inp, 'val.en')
    #dataset_save(val_tar, 'val.phone')
    #dataset_save(test_inp, 'test.en')
    #dataset_save(test_tar, 'test.phone')
    print()

def main():
    print()
    data = file_retrieve('cmudict-0.7b', 'ISO-8859-1')
    print('No. of lines in the original dataset: ', len(data))
    print()
    inp_lines, tar_lines = create_dataset(data)
    print('No. of input lines in the cleaned dataset: ', len(inp_lines))
    print('No. of target lines in the cleaned dataset: ', len(tar_lines))
    print()
    inp_lines, tar_lines = drop_duplicates(inp_lines, tar_lines)
    print('No. of input lines in the cleaned dataset: ', len(inp_lines))
    print('No. of target lines in the cleaned dataset: ', len(tar_lines))
    print()
    dataset_convert(inp_lines, tar_lines)

main()
