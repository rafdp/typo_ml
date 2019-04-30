import numpy as np
import tensorflow as tf
from load_hparams import *
from tensor2tensor.data_generators import text_encoder as te

from tensorflow.python.layers import core as layers_core

CheckHparamsMandatory("experiment")
CheckHparamsMandatory("data_path")
CheckHparamsMandatory("embedding_size")
CheckHparamsMandatory("source_lang")
CheckHparamsMandatory("target_lang")
CheckHparamsMandatory("learning_rate")
CheckHparamsMandatory("hidden_size")
CheckHparamsMandatory("num_layers")
CheckHparamsMandatory("input_encoder")
experiment = hparams["experiment"]
exp_path = hparams["data_path"] + experiment + '/'
source_lang = hparams["source_lang"]
target_lang = hparams["target_lang"]

corpus_info = np.load (exp_path + "dg/corpus_info.npy")
corpus_info = corpus_info[0]

train_source = np.load (exp_path + "dg/" + 'train_source.npy')
train_target = np.load (exp_path + "dg/" + 'train_target.npy')
evals_source = []
evals_target = []
for ev in corpus_info['evals']:
    evals_source.append(np.load (ev['save_path']+'_source.npy'))
    evals_target.append(np.load (ev['save_path']+'_target.npy'))
predict_source = np.load (exp_path + "dg/" + 'predict_source.npy')
predict_target = np.load (exp_path + "dg/" + 'predict_target.npy')

tf.logging.set_verbosity(tf.logging.INFO)

max_seq_len = 200
if "max_seq_len" in hparams:
    max_seq_len = hparams['max_seq_len']

num_epochs = 5
if "num_epochs" in hparams:
    num_epochs = hparams["num_epochs"]

batch_size = 20000
if "batch_size" in hparams:
    batch_size = hparams["batch_size"]

def train_generator():
    sample_count = 0 
    so_data = train_source
    a_data = train_target
    while sample_count < len(so_data):
        # count words and build batch
        max_len_in = 0 
        max_len_out = 0 
        if sample_count == len(so_data) - 1:
            break
        for i in range(sample_count, len(so_data)):
            max_len_in = max(max_len_in, np.sum(np.sign(so_data[i])))
            max_len_out = max(max_len_out, np.sum(np.sign(a_data[i])))
            max_len = max_len_in + max_len_out
            words_count = (i - sample_count + 1) * max_len
            if words_count > batch_size:
                break
        batch_so = so_data[sample_count: i]
        batch_a = a_data[sample_count: i]
        batch_features = {'features': batch_so}
        sample_count = i 
        yield batch_features, batch_a

def train_input_fn():
    types_dict = {'features': tf.int32}
    shapes_dict = {'features': [None, None]}
    dataset = tf.data.Dataset.from_generator(generator = lambda:train_generator(),
            output_types=(types_dict, tf.int32),
            output_shapes=(shapes_dict, [None, None]))
    dataset = dataset.batch(1)
    return dataset

eval_input    = [tf.estimator.inputs.numpy_input_fn(
                 x = {"features": evals_source[i]},
                 y = evals_target[i],
                 num_epochs = 1,
                 batch_size = 500, shuffle = False) 
                 for i in range(len(evals_source))]

predict_input = tf.estimator.inputs.numpy_input_fn(
                 x = {"features": predict_source},
                 y = predict_target,
                 num_epochs = 1,
                 batch_size = predict_source.size, shuffle = False)

def VocabSize(path):
    if not (os.path.isfile(path)):
        raise ValueError("Vocab not found (%s)" % path)
    return sum([1 for l in open(path)])
    

input_vocab_path = exp_path + source_lang + \
			    '(' + target_lang + ')' + '_vocab_' + hparams['input_encoder'] + '.txt'
input_vocab_size = VocabSize(input_vocab_path)

output_vocab_path = exp_path + '(' + source_lang + ')' + \
			     target_lang + '_vocab.txt'

output_vocab_size = VocabSize(output_vocab_path)

max_seq_len = 200
if "max_seq_len" in hparams:
    max_seq_len = hparams['max_seq_len']
        
def model_fn(features, labels, mode):
    
    source = tf.cast(features["features"], tf.int32, name = 'features')
    features_so = tf.reshape(source, [-1, tf.shape(source)[-1]])
        
    labels = tf.cast(labels, tf.int32, name = 'labels')
    source_embeddings = tf.get_variable('source_embeddings', 
                                        [input_vocab_size, 
                                         hparams['embedding_size']],
                                        initializer = tf.initializers.random_uniform)
    embedded_source = tf.nn.embedding_lookup(source_embeddings, features_so)
    batch_size_nn = tf.shape(features_so)[0]
    label_embeddings = tf.get_variable('answer_embeddings',
                                       [output_vocab_size, 
                                        hparams['embedding_size']],
                                       initializer = tf.initializers.random_uniform)
    labels = tf.cast(tf.reshape(labels, [-1, tf.shape(labels)[-1]]), tf.int32)

    embedded_labels = tf.nn.embedding_lookup(label_embeddings, labels)
    dropout_rate = .1
    if 'dropout_rate' in hparams:
        dropout_rate = hparams['dropout_rate']
    encoder_layers = [tf.contrib.rnn.GRUCell(hparams['hidden_size'], 
                                  name='enc_cell_%d' % i,
                                  activation=tf.nn.tanh) 
                     for i in range(hparams['num_layers'])]
    source_len = tf.sign(features_so)
    source_len = tf.reduce_sum(source_len, 1, name='source_len')
    source_len = tf.cast(source_len, tf.int32)
    enc_cell = tf.nn.rnn_cell.MultiRNNCell(encoder_layers)
    context, encoder_state = tf.nn.dynamic_rnn(enc_cell,
            embedded_source,
            dtype=tf.float32,
            sequence_length=source_len + 1)
    projection_layer = layers_core.Dense(output_vocab_size,
                                         use_bias=False, activation = tf.nn.softmax)
    used_labs = tf.sign(labels)
    length_labs = tf.reduce_sum(used_labs, 1, name='labels_length')
    length_labs = tf.cast(length_labs, tf.int32)
    max_ans_len = tf.reduce_max(length_labs)
    labs = tf.cast(labels[:,:max_ans_len], tf.int32)
    answer_mask = tf.sign(labs, name='answer_mask')
    answer_mask = tf.cast(answer_mask, tf.float32)
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(hparams['hidden_size'], 
                                                             memory = context,
                                                             memory_sequence_length = source_len + 1)
    """    
    decoder_layers = [tf.contrib.seq2seq.AttentionWrapper(
                      tf.contrib.rnn.GRUCell (hparams['hidden_size'], 
                                              name='dec_cell_%i' % i,
                                              activation=tf.nn.tanh),
                      attention_mechanism = attention_mechanism)
                      for i in range(hparams['num_layers'])]
    """
    decoder_layers = [tf.contrib.rnn.GRUCell (hparams['hidden_size'], 
                                              name='dec_cell_%i' % i,
                                              activation=tf.nn.tanh)
                      for i in range(hparams['num_layers'])]
    
    dec_cell = tf.nn.rnn_cell.MultiRNNCell(decoder_layers)
    #initial_state = tuple([decoder_layers[i].zero_state(batch_size_nn, tf.float32)\
    #                       .clone(cell_state = encoder_state[i])\
    #                for i in range(hparams['num_layers'])])
    initial_state = encoder_state
    #initial_state = dec_cell.zero_state(batch_size_nn, tf.float32)
    #initial_state1 = tf.nn.rnn_cell.MultiRNNCell(decoder_layers1).zero_state(batch_size_nn, tf.int32)

    temp = tf.identity(tf.shape(source_len), 'temp1')
    helper = tf.contrib.seq2seq.TrainingHelper(
                embedded_labels, length_labs, time_major=False)
    decoder = tf.contrib.seq2seq.BasicDecoder(
                dec_cell, helper, initial_state,
                output_layer=projection_layer)
    outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder)

    classes = outputs.sample_id
    probabilities = outputs.rnn_output

    answer_ref = tf.one_hot(labs, output_vocab_size)
    cross_entropy = tf.log(probabilities + 1e-8) * answer_ref
    cross_entropy = tf.identity(-tf.reduce_sum(cross_entropy, axis=2), 'cross_entropy')
    answer_mask = tf.stop_gradient(answer_mask)
    cross_entropy *= answer_mask
    cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)

    ans_norm = tf.reshape(tf.reduce_sum(answer_mask, 1), [-1, 1]) + 0.1
    cross_entropy /= tf.cast(ans_norm, tf.float32)

    loss_ce = tf.reduce_mean(cross_entropy)
    loss = tf.identity(loss_ce, name='loss')
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=hparams['learning_rate'])

        train_op = optimizer.minimize(
                   loss=loss,
                    global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,                    
                        predictions=None,
                        train_op = train_op,
                        loss = loss)
    eval_metric_ops = {}
    eval_metric_ops["accuracy_words"] = tf.metrics.accuracy(
                labels=labs, predictions=classes)
    return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)        


cycle_secs = 300
if 'cycle_secs' in hparams:
    cycle_secs = hparams['cycle_secs'] 
chkp_config = tf.estimator.RunConfig(save_checkpoints_secs = cycle_secs)

tensors_to_log = {'loss': 'loss', 'temp1': 'temp1'}
logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=10)

print("starting epoch")
estimator = tf.estimator.Estimator(                                     
            model_fn=model_fn, model_dir=exp_path+'model_dir', config = chkp_config)


for i in range(num_epochs):
    estimator.train(train_input_fn, [logging_hook])
    for j in range(len(evals_source)):
        estimator.evaluate(eval_input[j], name = corpus_info['evals'][j]['name'])

                                             
print("END")
    

