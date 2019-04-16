
from tensor2tensor.data_generators import text_encoder as te
import translate.storage.tmx as tmx
from load_hparams import *
import numpy as np
import os
import operator
import string

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
    return (corpus, cut, len(cum_sum))

source_corpus, source_cut, source_count = Normalize(source_corpus, removal_rate)
print("    input: removed %d of %d characters" % (source_count - source_cut, source_count))
target_corpus, target_cut, target_count = Normalize(target_corpus, removal_rate)
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
######### Encoding
######################

print("Encoding:")
encoded_source = np.array([input_encoder.encode(utt) 
                           for utt in source_corpus])

encoded_target = np.array([output_encoder.encode(utt) 
                           for utt in target_corpus])
sz = encoded_source.size
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

