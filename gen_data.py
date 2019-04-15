
from tensor2tensor.data_generators import text_encoder as te
import translate.storage.tmx as tmx
from load_hparams import *
import numpy as np
import os



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
source_corpus = [unit.getNodeText(unit.getlanguageNode(source_lang)) 
                 for unit in units]
target_corpus = [unit.getNodeText(unit.getlanguageNode(target_lang)) 
                 for unit in units]



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
        print("    building..")
        encoder = te.SubwordTextEncoder.\
                        build_from_generator(corpus,
                                   target_size=hparams['vocab_size'])
        print("    storing..")
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

