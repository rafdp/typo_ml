
from tensor2tensor.data_generators import text_encoder as te
import translate.storage.tmx as tmx
from load_hparams import *
import numpy as np
import os
import operator
import string
import random
CheckHparamsMandatory("source_lang")
CheckHparamsMandatory("target_lang")
CheckHparamsMandatory("data_path")
CheckHparamsMandatory("vocab_size")
CheckHparamsMandatory("experiment")

experiment = hparams["experiment"]

source_lang = hparams["source_lang"]
target_lang = hparams["target_lang"]

exp_path = hparams["data_path"] + experiment + '/'
if not os.path.isdir(exp_path + 'dg'):
    os.makedirs(exp_path + 'dg')

######################
######### Corpus work
######################

print("Corpus:")
print("  loading ", end='', flush=True)
with open(hparams["data_path"] + source_lang + '-' + target_lang + '.tmx', "rb") as f:
    corpus = tmx.tmxfile(f, source_lang, target_lang)
units = corpus.getunits()
print(len(units))
print("  rearranging")
source_corpus = np.array([unit.getNodeText(unit.getlanguageNode(source_lang)) 
                          for unit in units])
target_corpus = np.array([unit.getNodeText(unit.getlanguageNode(target_lang)) 
                          for unit in units])
for i in reversed(range(len(source_corpus))):
    if len(source_corpus[i]) < 1 or len(target_corpus[i]) < 1:
        del source_corpus[i]
        del target_corpus[i]
        i -= 1

removal_rate = 0.05
if "char_removal%" in hparams:
    removal_rate = hparams["char_removal%"]/100.0

print("  normalizing (char removal rate %d %%)" % int(removal_rate*100))

def Normalize(corpus, rate):
    chars = {}
    
    for i in range(len(corpus)):
        corpus[i] = corpus[i].lower().translate(str.maketrans('', '', string.punctuation))
        for c in corpus[i]:
            if c not in chars:
                chars[c] = 1
            else :
                chars[c] += 1
    
    sorted_chars = sorted(chars.items(), key=operator.itemgetter(1))
    cum_sum = []
    prev_sum = 0
    for i in range(len(sorted_chars)):
        cum_sum.append(prev_sum + chars[sorted_chars[i][0]])
        prev_sum = cum_sum[i]
    cut = 0
    for i in range(len(cum_sum)):
        if cum_sum[i] > (1-rate)*prev_sum:
            cut = i
            break
    delete_chars = ''.join([c[0] for c in sorted_chars[cut:]])
    
    for s in corpus:
        s = s.replace(delete_chars, '')
    return (corpus, cut, len(cum_sum), sorted_chars[:cut])

source_corpus, source_cut, source_count, source_alph = Normalize(source_corpus, removal_rate)
print("    input: removed %d of %d characters" % (source_count - source_cut, source_count))
target_corpus, target_cut, target_count, _ = Normalize(target_corpus, removal_rate)
print("    output: removed %d of %d characters" % (target_count - target_cut, source_count))

print("  shuffling")
rand_state = np.random.get_state()
np.random.shuffle(source_corpus)
np.random.set_state(rand_state)
np.random.shuffle(target_corpus)

print("Test strings:")
print("  input: '%s'" % source_corpus[0])
print("  output: '%s'" % target_corpus[0])

######################
######### Vocab work
######################

######### Input

def GenerateVocab(path, corpus):

    if os.path.isfile(path) and \
       ("rebuild_vocab" not in hparams or not hparams["rebuild_vocab"]):
        print("    vocab found")
        return te.SubwordTextEncoder(path)
    else:
        print("    building")
        encoder = te.SubwordTextEncoder.\
                        build_from_generator(corpus,
                                   target_size=hparams['vocab_size'])
        print("    storing")
        encoder.store_to_file(path)
        return encoder
    

print("Building encoders:")

print("  input:")
input_vocab_path = exp_path + source_lang + \
                            '(' + target_lang + ')' + '_vocab.txt'
input_encoder = GenerateVocab(input_vocab_path, source_corpus)

print("  output:")
output_vocab_path = exp_path + '(' + source_lang + ')' + \
                             target_lang + '_vocab.txt'
output_encoder = GenerateVocab(output_vocab_path, target_corpus)
######################
######### Typo work
#####################

sz = len(source_corpus)
eval_pc = 0.1
if "eval%" in hparams: 
    eval_pc = hparams["eval%"] / 100
eval_sz = int(sz * eval_pc)

predict_pc = 0.1
if "predict%" in hparams: 
    predict_pc = hparams["predict%"] / 100
predict_sz = int(sz*predict_pc)
train_sz = sz - eval_sz - predict_sz
if train_sz < 0:
    raise ValueError("Invalid sizes for eval and predict")

def ApplyTypoInjectorByIndex(start_idx, end_idx, func, prob):
    for i in range(start_idx, end_idx):
        if len(source_corpus[i]) < 4:
            continue
        char_idx = 0
        rand_val = random.random()
        if rand_val > prob:
            continue
        indices = []
        while char_idx < len(source_corpus[i]):
            while char_idx < len(source_corpus[i]) and \
                  source_corpus[i][char_idx] == ' ': 
                char_idx += 1
            if char_idx >= len(source_corpus[i]):
                break
            next_char_idx = char_idx + 1
            while next_char_idx < len(source_corpus[i]) and \
                  source_corpus[i][next_char_idx] != ' ': 
                next_char_idx += 1
            if next_char_idx - char_idx < 4:
                char_idx = next_char_idx + 1
                continue
            indices.append([char_idx, next_char_idx])
            char_idx = next_char_idx + 1
        if len(indices) == 0:
            indices.append([0, len(source_corpus[i])])
        manip = indices[random.randint(0, len(indices)-1)]
        func(i, manip[0], manip[1])

def ApplyTypoInjectorPhrase(start_idx, end_idx, func, prob):
    for i in range(start_idx, end_idx):
        rand_val = random.random()
        if rand_val < prob:
            func(i, prob)

def InjectGlue(i, prob):
    print('  injecting glue:')
    print('    %s' % source_corpus[i])
    indices = []
    for j in range(len(source_corpus[i])):
        if source_corpus[i][j] == ' ':
            indices.append(j)
    random.shuffle(indices) 
    rand_val = 1
    indices_new = []
    for j in range(len(indices)):
        if rand_val < 1 - prob:
            break
        indices_new.append(indices[j])
        rand_val *= random.random()
    indices_new = sorted(indices_new)
    for j in reversed(range(len(indices_new))):
        source_corpus[i] = source_corpus[i][:indices_new[j]] + \
                           source_corpus[i][indices_new[j]+1:] 
    print('    %s' % source_corpus[i])

def InjectSwap(i, char_idx, next_char_idx):
    print('  injecting swap:')
    print('    %s' % source_corpus[i][char_idx:next_char_idx])
    idx = random.randint(0, next_char_idx - char_idx - 2)
    c = source_corpus[i][char_idx + idx]
    source_corpus[i] = source_corpus[i][:char_idx + idx] + \
                       source_corpus[i][char_idx+idx+1] + \
                       source_corpus[i][char_idx + idx+1:] 
    source_corpus[i] = source_corpus[i][:char_idx + idx+1] + \
                       c + \
                       source_corpus[i][char_idx + idx+2:] 
    print('    %s' % source_corpus[i][char_idx:next_char_idx])
    return 0

def InjectRandom(i, char_idx, next_char_idx):
    print('  injecting random:')
    print('    %s' % source_corpus[i][char_idx:next_char_idx])
    idx = random.randint(0, next_char_idx - char_idx)
    letter = source_alph[random.randint(0, len(source_alph) - 1)][0]
    source_corpus[i] = source_corpus[i][:char_idx] + letter + \
                       source_corpus[i][char_idx+1:]
    print('    %s' % source_corpus[i][char_idx:next_char_idx])
    return 0
 
def InjectInsert(i, char_idx, next_char_idx):
    print('  injecting insert:')
    print('    %s' % source_corpus[i][char_idx:next_char_idx])
    idx = random.randint(0, next_char_idx - char_idx)
    letter = source_alph[random.randint(0, len(source_alph)-1)][0]
    source_corpus[i] = source_corpus[i][:char_idx+idx] + letter + \
                       source_corpus[i][char_idx+idx:]
    print('    %s' % source_corpus[i][char_idx:next_char_idx+1])
    return 1
   
def InjectSplit(i, char_idx, next_char_idx):
    print('  injecting split:')
    print('    %s' % source_corpus[i][char_idx:next_char_idx])
    idx = random.randint(1, next_char_idx - char_idx-1)
    source_corpus[i] = source_corpus[i][:char_idx+idx] + ' ' + \
                       source_corpus[i][char_idx+idx:]
    print('    %s' % source_corpus[i][char_idx:next_char_idx + 1])
    return 1

def InjectDelete(i, char_idx, next_char_idx):
    print('  injecting delete:')
    print('    %s' % source_corpus[i][char_idx:next_char_idx])
    idx = random.randint(0, next_char_idx - char_idx)
    source_corpus[i] = source_corpus[i][:char_idx+idx] + \
                       source_corpus[i][char_idx+idx + 1:]
    print('    %s' % source_corpus[i][char_idx:next_char_idx-1])
    return -1
def InjectDouble(i, char_idx, next_char_idx):
    print('  injecting double:')
    print('    %s' % source_corpus[i][char_idx:next_char_idx])
    idx = random.randint(0, next_char_idx - char_idx - 1)
    source_corpus[i] = source_corpus[i][:char_idx+idx] + \
                       source_corpus[i][char_idx + idx] + \
                       source_corpus[i][char_idx+idx:]
    print('    %s' % source_corpus[i][char_idx:next_char_idx+1])
    return 1
            
print("Injecting typos")
typo_indices = {"eval": [train_sz, train_sz + eval_sz],
                "predict": [train_sz + eval_sz, train_sz + eval_sz + predict_sz]}
injectFuncs = {"split":  InjectSplit,
               "swap":   InjectSwap,
               "random": InjectRandom,
               "delete": InjectDelete,
               "insert": InjectInsert,
               "double": InjectDouble}

for typo in hparams['inject_typos']:
    print('  %s:' % typo)
    if typo == "glue":
        for stage in hparams['inject_typos'][typo]:
            ApplyTypoInjectorPhrase(
                 typo_indices[stage][0],
                 typo_indices[stage][1],
                 InjectGlue,
                 hparams['inject_typos'][typo][stage]/100)
    elif typo in injectFuncs:
        for stage in hparams['inject_typos'][typo]:
            ApplyTypoInjectorByIndex(
                 typo_indices[stage][0],
                 typo_indices[stage][1],
                 injectFuncs[typo],
                 hparams['inject_typos'][typo][stage]/100)
    else:
        raise ValueError("Invalid typo type. Very funny. (%s)" % typo) 

######################
######### Encoding
######################

print("Encoding:")

max_seq_len = 200
if "max_seq_len" in hparams:
    max_seq_len = hparams['max_seq_len']
encoded_source = np.zeros((sz, max_seq_len), dtype = int)
for i in range(sz):
    encoded = input_encoder.encode(source_corpus[i])
    if len(encoded) > max_seq_len:
        raise BaseException("max_seq_len must be bigger than utterance length")
    for x in range(len(encoded)):
        encoded_source[i][x] = encoded[x]

encoded_target = np.zeros((sz, max_seq_len), dtype = int)
for i in range(sz):
    encoded = output_encoder.encode(target_corpus[i])
    if len(encoded) > max_seq_len:
        raise BaseException("max_seq_len must be bigger than utterance length")
    for x in range(len(encoded)):
        encoded_target[i][x] = encoded[x]


print("  train %d (%.1f %%)" % (train_sz, train_sz / sz*100))
train_source   = encoded_source[ : train_sz]
train_target   = encoded_target[ : train_sz]

print("  eval %d (%.1f %%)" % (eval_sz, eval_sz / sz * 100))
eval_source    = encoded_source[train_sz : train_sz + eval_sz]
eval_target    = encoded_target[train_sz : train_sz + eval_sz]

print("  predict %d (%.1f %%)" % (predict_sz, predict_sz / sz * 100))
predict_source = encoded_source[train_sz + eval_sz : ]
predict_target = encoded_target[train_sz + eval_sz : ]

print("  storing")
np.save (exp_path + "dg/" + 'train_source', train_source)
np.save (exp_path + "dg/" + 'train_target', train_target)
np.save (exp_path + "dg/" + 'eval_source', eval_source)
np.save (exp_path + "dg/" + 'eval_target', eval_target)
np.save (exp_path + "dg/" + 'predict_source', predict_source)
np.save (exp_path + "dg/" + 'predict_target', predict_target)

