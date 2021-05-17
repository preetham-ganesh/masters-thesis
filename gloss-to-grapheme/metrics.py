import numpy as np
import math
import operator
from functools import reduce
from nltk.translate.meteor_score import meteor_score
import pandas as pd

def count_ngram(candidate, references, n):
    clipped_count = 0
    count = 0
    r = 0
    c = 0
    for si in range(len(candidate)):
        ref_counts = []
        ref_lengths = []
        for reference in references:
            ref_sentence = reference[si]
            ngram_d = {}
            words = ref_sentence.strip().split()
            ref_lengths.append(len(words))
            limits = len(words) - n + 1
            for i in range(limits):
                ngram = ' '.join(words[i:i+n]).lower()
                if ngram in ngram_d.keys():
                    ngram_d[ngram] += 1
                else:
                    ngram_d[ngram] = 1
            ref_counts.append(ngram_d)
        cand_sentence = candidate[si]
        cand_dict = {}
        words = cand_sentence.strip().split()
        limits = len(words) - n + 1
        for i in range(0, limits):
            ngram = ' '.join(words[i:i + n]).lower()
            if ngram in cand_dict:
                cand_dict[ngram] += 1
            else:
                cand_dict[ngram] = 1
        clipped_count += clip_count(cand_dict, ref_counts)
        count += limits
        r += best_length_match(ref_lengths, len(words))
        c += len(words)
    if clipped_count == 0:
        pr = 0
    else:
        pr = float(clipped_count) / count
    bp = brevity_penalty(c, r)
    return pr, bp

def clip_count(cand_d, ref_ds):
    count = 0
    for m in cand_d.keys():
        m_w = cand_d[m]
        m_max = 0
        for ref in ref_ds:
            if m in ref:
                m_max = max(m_max, ref[m])
        m_w = min(m_w, m_max)
        count += m_w
    return count

def best_length_match(ref_l, cand_l):
    least_diff = abs(cand_l-ref_l[0])
    best = ref_l[0]
    for ref in ref_l:
        if abs(cand_l-ref) < least_diff:
            least_diff = abs(cand_l-ref)
            best = ref
    return best

def brevity_penalty(c, r):
    if c > r:
        bp = 1
    else:
        bp = math.exp(1-(float(r)/c))
    return bp

def geometric_mean(precisions):
    return (reduce(operator.mul, precisions)) ** (1.0 / len(precisions))

def meteor(candidate, reference):
    score = []
    for i, j in zip(candidate, reference):
        s = meteor_score([j], i, 4)
        score.append(s)
    return np.mean(score)

def BLEU(candidate, references):
    precisions = []
    for i in range(4):
        pr, bp = count_ngram(candidate, references, i+1)
        precisions.append(pr)
    bleu = geometric_mean(precisions) * bp
    return bleu, geometric_mean(precisions), bp

def edit_distance(r, h):
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8).reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        d[i][0] = i
    for j in range(len(h)+1):
        d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                delete = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return d

def text_retrieve(name):
    with open('/home/preetham/Documents/Preetham/masters-thesis/results/gloss-to-grapheme/'+name, 'r', encoding='utf-8') as f:
        text = f.read()
    f.close()
    return text.split('\n')

def wer_score(r, h):
    wer_l = []
    for i, j in zip(r, h):
        i = i.split()
        j = j.split()
        d = edit_distance(i, j)
        wer = float(d[len(i)][len(j)] * 100) / len(i)
        wer_l.append(wer)
    return np.mean(wer_l)

def create_sub_dataset(inp, tar, pred, range_0, range_1):
    new_tar, new_pred = [], []
    for i, j, k in zip(tar, pred, inp):
        if len(k.split(' ')) >= range_0 and len(k.split(' ')) <= range_1:
            new_tar.append(i)
            new_pred.append(j)
    return new_tar, new_pred

def complete_metrics(model, model_type, file_name):
    metrics = ['model', 'config', 'wer', 'bleu', 'ps', 'meteor']
    d = {i:[] for i in metrics}
    for i in model:
        for j in model_type:
            inp = text_retrieve(j + '/model_' + str(i) + '/predictions/' + file_name + '_inp.txt')
            tar = text_retrieve(j + '/model_' + str(i) + '/predictions/' + file_name + '_tar.txt')
            pred = text_retrieve(j + '/model_' + str(i) + '/predictions/' + file_name + '_pred.txt')
            wer = wer_score(tar, pred)
            bleu, ps, bp = BLEU(pred, [tar])
            met = meteor(pred, tar)
            if i == 1:
                d['model'].append(j)
            else:
                d['model'].append('')
            d['config'].append(j[0]+'-'+str(i))
            d['wer'].append(round(wer, 2))
            d['bleu'].append(round(bleu*100, 2))
            d['ps'].append(round(ps*100, 2))
            d['meteor'].append(round(met*100, 2))
    df = pd.DataFrame(d, columns=metrics)
    print(file_name + ' complete metrics calculated')
    print()
    df.to_csv('/home/preetham/Documents/Preetham/masters-thesis/results/gloss-to-grapheme/metrics/'+file_name+'-full.csv',
              index=False)

def length_metrics(model, model_type, file_name):
    metrics = ['model', 'config', 'wer_10', 'bleu_10', 'ps_10', 'meteor_10', 'wer_20', 'bleu_20', 'ps_20',
               'meteor_20', 'wer_30', 'bleu_30', 'ps_30', 'meteor_30', 'wer_40', 'bleu_40', 'ps_40', 'meteor_40']
    ranges = [[0, 10], [11, 20], [21, 30], [31, 40]]
    d = {i: [] for i in metrics}
    for i in model:
        for j in model_type:
            inp = text_retrieve(j + '/model_' + str(i) + '/predictions/' + file_name + '_inp.txt')
            tar = text_retrieve(j + '/model_' + str(i) + '/predictions/' + file_name + '_tar.txt')
            pred = text_retrieve(j + '/model_' + str(i) + '/predictions/' + file_name + '_pred.txt')
            if i == 1:
                d['model'].append(j)
            else:
                d['model'].append('')
            d['config'].append(j[0] + '-' + str(i))
            for k in ranges:
                new_tar, new_pred = create_sub_dataset(inp, tar, pred, k[0], k[1])
                wer = wer_score(new_tar, new_pred)
                bleu, ps, bp = BLEU(new_pred, [new_tar])
                met = meteor(new_pred, new_tar)
                d['wer_'+str(k[1])].append(round(wer, 2))
                d['bleu_'+str(k[1])].append(round(bleu * 100, 2))
                d['ps_'+str(k[1])].append(round(ps * 100, 2))
                d['meteor_'+str(k[1])].append(round(met * 100, 2))
    df = pd.DataFrame(d, columns=metrics)
    print(file_name + ' length metrics calculated')
    print()
    df.to_csv(
        '/home/preetham/Documents/Preetham/masters-thesis/results/gloss-to-grapheme/metrics/' + file_name + '-length.csv',
        index=False)

def main():
    print()
    model = [i for i in range(1, 9)]
    model_type = ['bahdanau', 'luong', 'transformer']
    complete_metrics(model, model_type, 'val')
    complete_metrics(model, model_type, 'test')
    length_metrics(model, model_type, 'val')
    length_metrics(model, model_type, 'test')

main()