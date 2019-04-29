import sys
import json
import getpass
import os

def replace_names(where, name, target):
    if type(where) == str and name in where:
        return where.replace(name, target)
    if type(where) == dict:
        for val in where:
            where[val] = replace_names(where[val], name, target)
        return where
    if type(where) == list:
        for val_idx in range(len(where)):
            where[val_idx] = replace_names(where[val_idx], name, target)
        return where
    return where

def loader_func(hparams_path, is_parent=False):
    hparams = json.load(open(hparams_path))
    
        # replace all <smth> substrings to hparams["smth"]
    for p1 in hparams:
        hparams = replace_names(hparams, '<' + p1 + '>', hparams[p1])
    
    return hparams

try:
    hparams = loader_func(sys.argv[1])
except BaseException as e:
    print('\nERROR: valid hparams file must be specified as first argument\n')
    hparams = loader_func(sys.argv[1])

def CheckHparamsMandatory(key, subsection = None):
    subhparams = hparams
    print_subsection = False
    if subsection != None:
        subhparams = hparams[subsection]
        print_subsection = True
    if not key in subhparams:
        if print_subsection:
            raise NameError("Mandatory key %s in subsection %s not found in hparams" % (key, subsection))
        else: 
            raise NameError("Mandatory key %s not found in hparams" % key)


"""
def PrintHparamsInfo(hparams):
    def bool_to_str(val):
        if val:
            return 'true'
        else:
            return 'false'
    ESCAPE_INFO = '\033[38;5;209m'
    ESCAPE_TITLE = '\033[38;5;123m'
    ESCAPE_DATA = '\033[38;5;72m'
    ESCAPE_FILE = '\033[38;5;118m'
    ESCAPE_OFF = '\033[0m'
    import __main__
    print('\n\n\n')
    print(ESCAPE_TITLE + 'Running ' + ESCAPE_FILE +  __main__.__file__ + ESCAPE_TITLE + ' on model ' + ESCAPE_INFO + hparams['model_name'])
    print(ESCAPE_INFO + '    input encoders: ')
    for enc in hparams['input_encoders']:    
        print(ESCAPE_INFO + '      type: ' + ESCAPE_DATA + enc['type'])
        if (enc['bag']):
            print(ESCAPE_DATA + '        bags')
        if (enc['positional_encoding']):
            print(ESCAPE_DATA + '        pe')
        if (enc['build']):
            print(ESCAPE_DATA + '        build')
    print(ESCAPE_INFO + '    output encoder: ')
    print(ESCAPE_INFO + '      type: ' + ESCAPE_DATA + hparams['output_encoder']['type'])
    print(ESCAPE_INFO + '        build: ' + ESCAPE_DATA + bool_to_str(hparams['output_encoder']['build']))
    print(ESCAPE_OFF + '\n\n\n')
"""
