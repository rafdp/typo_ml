import numpy as np
import tensorflow as tf
from load_hparams import *
from tensor2tensor.data_generators import text_encoder as te
from tensor2tensor.utils import bleu_hook
from tensorflow.python.layers import core as layers_core
from tensorflow.python.util import nest
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import tensor_shape


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

print(hparams['learning_rate'])

corpus_info = np.load (exp_path + "dg/corpus_info.npy")
corpus_info = corpus_info[0]
#corpus_info['evals'] = corpus_info['evals'][:1]
#train_source = np.repeat(np.load (exp_path + "dg/" + 'train_source.npy')[:10], 100000, axis=0)
train_source = np.load (exp_path + "dg/" + 'train_source.npy')
print(train_source.shape)
#train_target = np.repeat(np.load (exp_path + "dg/" + 'train_target.npy')[:10], 100000, axis=0)
train_target = np.load (exp_path + "dg/" + 'train_target.npy')
evals_source = []
evals_target = []

#eval_limiting = 10000
for ev in corpus_info['evals']:
    evals_source.append(np.load (ev['save_path']+'_source.npy'))
    evals_target.append(np.load (ev['save_path']+'_target.npy'))
    print("eval %s, sizes %d %d" % (ev["save_path"] + '_source.npy', len(evals_source[-1]), len(evals_target[-1])))


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

corpus_info["input_vocab_size"] = input_vocab_size
corpus_info["output_vocab_size"] = output_vocab_size

def model_fn(features, labels, mode):
    
    source = tf.cast(features["features"], tf.int32, name = 'features')
    features_so = tf.reshape(source, [-1, tf.shape(source)[-1]])
    source_len = tf.sign(features_so)
    source_len = tf.reduce_sum(source_len, 1, name='source_len')
    source_len = tf.cast(source_len, tf.int32)

    max_source_len = tf.reduce_max([tf.reduce_max(source_len, name='max_len'),  2])
    features_so = features_so[:, :max_source_len]
    #stripped_shape = tf.shape(stripped_features_so, name='stripped_shape')
    if mode != tf.estimator.ModeKeys.PREDICT:
        labels = tf.cast(labels, tf.int32, name = 'labels')
    source_embeddings = tf.get_variable('source_embeddings', 
                                        [input_vocab_size, 
                                         hparams['embedding_size']],
                                        initializer = tf.initializers.random_uniform)
    distr = tf.identity(features_so)
    source_mask = tf.cast(tf.sign(features_so, name='source_mask'), tf.float32)
    embedded_source = tf.nn.embedding_lookup(source_embeddings, features_so)
    batch_size_nn = tf.shape(features_so)[0]
    label_embeddings = tf.get_variable('answer_embeddings',
                                       [output_vocab_size, 
                                        hparams['embedding_size']],
                                       initializer = tf.initializers.random_uniform)
    if mode != tf.estimator.ModeKeys.PREDICT:
        labels = tf.cast(tf.reshape(labels, [-1, tf.shape(labels)[-1]]), tf.int32)

        embedded_labels = tf.nn.embedding_lookup(label_embeddings, labels)
    dropout_rate = .1
    if 'dropout_rate' in hparams:
        dropout_rate = hparams['dropout_rate']
    encoder_layers = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(hparams['hidden_size'], 
                                  name='enc_cell_%d' % i,
                                  activation=tf.nn.tanh), dropout_rate)
                     for i in range(hparams['num_layers'])]


    enc_cell = tf.nn.rnn_cell.MultiRNNCell(encoder_layers)
    context, encoder_state = tf.nn.dynamic_rnn(enc_cell,
            embedded_source,
            dtype=tf.float32,
            sequence_length=source_len + 1)
    projection_layer = layers_core.Dense(output_vocab_size,
                                         use_bias=False,
                                         activation = tf.nn.softmax)
    if mode != tf.estimator.ModeKeys.PREDICT:
        used_labs = tf.sign(labels)
        length_labs = tf.reduce_sum(used_labs, 1, name='labels_length')
        length_labs = tf.cast(length_labs, tf.int32)
        max_ans_len = tf.reduce_max(length_labs)
        labs = tf.cast(labels[:,:max_ans_len], tf.int32)
        answer_mask = tf.sign(labs, name='answer_mask')
        answer_mask = tf.cast(answer_mask, tf.float32)
    else:
        beam_width = 3
        if 'beam_width' in hparams:
            beam_width = hparams['beam_width']
        context = tf.contrib.seq2seq.tile_batch(
                context, multiplier=beam_width)
        distr = tf.contrib.seq2seq.tile_batch(distr, multiplier=beam_width)
        source_mask = tf.contrib.seq2seq.tile_batch(source_mask, multiplier=beam_width)
        encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier = beam_width)
        source_len = tf.contrib.seq2seq.tile_batch(source_len, multiplier = beam_width)

    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(hparams['hidden_size'], 
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
    
    #dec_cell = ModifiedGRUCell (hparams['hidden_size'], activation = tf.nn.tanh)
    decoder_layers[0] = tf.contrib.seq2seq.AttentionWrapper(decoder_layers[0], attention_mechanism)
    
    dec_cell = tf.nn.rnn_cell.MultiRNNCell(decoder_layers)
    batch_size_temp = batch_size_nn
    if mode != tf.estimator.ModeKeys.PREDICT:
        dec_init_state = encoder_state
    else:
        #pass
        batch_size_temp *= beam_width
        dec_init_state = tf.contrib.seq2seq.tile_batch(encoder_state[-1], multiplier = beam_width)
    initial_state = [encoder_state[-1] for i in range(hparams['num_layers'])]
    #initial_state = encoder_state
    attention_cell_state = decoder_layers[0].zero_state(dtype=tf.float32, batch_size=batch_size_temp)
    initial_state[0] = attention_cell_state.clone(cell_state=initial_state[0])
    initial_state = tuple(initial_state)
    #initial_state = dec_cell.zero_state(batch_size_nn, tf.float32)
    #initial_state1 = tf.nn.rnn_cell.MultiRNNCell(decoder_layers1).zero_state(batch_size_nn, tf.int32)
    #initial_state = encoder_state
    if mode == tf.estimator.ModeKeys.PREDICT:
        #initial_state = tf.contrib.seq2seq.tile_batch(
        #            initial_state, multiplier=beam_width)
        
        #projection_layer = layers_core.Dense(output_vocab_size,
        #                                 use_bias=False)
        
        decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                        cell=dec_cell,
                        embedding=label_embeddings,
                        start_tokens=tf.fill([batch_size_nn], 0),
                        end_token=1,
                        initial_state=initial_state,
                        beam_width=beam_width,
                        length_penalty_weight = 0.6,
                        output_layer=projection_layer)
        """
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                        label_embeddings,
                        tf.fill([batch_size_nn], 0), 1)
        
        decoder = tf.contrib.seq2seq.BasicDecoder(
                        dec_cell, helper, initial_state,
                        output_layer=projection_layer)
        """
        outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder, maximum_iterations=max_source_len)
        classes = outputs.predicted_ids[:, :, 0]
        #classes_topn = outputs.predicted_ids
        #beam_scores = outputs.beam_search_decoder_output.scores
        #classes = outputs.sample_id
        #att_image = list(final_state.cell_state)[0][:, 0, :]
        #print(final_state.cell_state)
        #att_image = tf.reshape(att_image,
        #                [-1, max_source_len, max_source_len])

        predictions = {}
        predictions['classes'] = classes
        #predictions['classes_topn'] = classes_topn
        #predictions['beam_scores'] = beam_scores
        #predictions['attention_image'] = a/t_image
        print('predictions:', predictions)
        return tf.estimator.EstimatorSpec(mode=mode,
                    predictions=predictions,
                    export_outputs={'output':tf.estimator.export.PredictOutput(classes)})

    helper = tf.contrib.seq2seq.TrainingHelper(
                embedded_labels, length_labs, time_major=False)
    decoder = tf.contrib.seq2seq.BasicDecoder(
                dec_cell, helper, initial_state,
                output_layer=projection_layer)
    outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
    classes = outputs.sample_id
    probabilities = outputs.rnn_output

    temp = tf.identity(tf.shape(probabilities), 'temp1')
    answer_ref = tf.one_hot(labs, output_vocab_size)
    cross_entropy = tf.log(probabilities + 1e-8) * answer_ref
    cross_entropy = tf.identity(-tf.reduce_sum(cross_entropy, axis=2), 'cross_entropy')
    answer_mask = tf.stop_gradient(answer_mask)
    cross_entropy *= answer_mask
    cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)

    ans_norm = tf.reshape(tf.reduce_sum(answer_mask, 1), [-1, 1]) + 0.1
    cross_entropy /= tf.cast(ans_norm, tf.float32)

    loss_ce = tf.reduce_mean(cross_entropy)
    #loss_ce = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(
    #                         logits=probabilities,
    #                         targets=labs,
    #                         weights=tf.to_float(tf.ones_like(labs)), average_across_batch=True))
    loss = tf.identity(loss_ce, name='loss')
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=hparams['learning_rate'])

        train_op = optimizer.minimize(
                   loss=loss_ce,
                    global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,                    
                        predictions=None,
                        train_op = train_op,
                        loss = loss_ce)
    eval_metric_ops = {}
    eval_metric_ops["accuracy_words"] = tf.metrics.accuracy(
                labels=labs, predictions=classes)
    return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)        

cycle_secs = 300
if 'cycle_secs' in hparams:
    cycle_secs = hparams['cycle_secs'] 
chkp_config = tf.estimator.RunConfig(save_checkpoints_secs = cycle_secs)

tensors_to_log = {'loss': 'loss'}
logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=10)

print("starting epoch")
estimator = tf.estimator.Estimator(                                     
            model_fn=model_fn, model_dir=exp_path+'model_dir', config = chkp_config)

output_encoder = te.SubwordTextEncoder(corpus_info['output_vocab_path'])
summary_writer = tf.summary.FileWriter(exp_path)
for i in range(num_epochs):
    estimator.train(train_input_fn, [logging_hook])
    if i % 3 != 2:
        continue
    for k in range(len(evals_source)):
        #print(estimator.evaluate(eval_input[j], name = corpus_info['evals'][j]['name']))
        results_prediction = estimator.predict(input_fn=eval_input[k])
        predictions = []
        for j, r in enumerate(results_prediction):
            if j % 1000 == 0:
                print('Predicting...', j)
            sent_vec = r['classes']
            if 1 in sent_vec:
                sent_vec = sent_vec[:list(sent_vec).index(1)]
            if 0 in sent_vec:
                sent_vec = sent_vec[:list(sent_vec).index(0)]
            sent = output_encoder.decode(sent_vec)
            predictions.append(sent)
        file_path = exp_path + ('predictions/prediction_%s.txt' % corpus_info['evals'][k]['name'])
        print(file_path)
        pred_tmp_file = open(file_path, 'w')
        for s in predictions:
            pred_tmp_file.write("%s\n" % s)
        pred_tmp_file.close()
        print("Comparing %s and %s" % (corpus_info['eval_target_path'], file_path))
        bleu = bleu_hook.bleu_wrapper(ref_filename=corpus_info['eval_target_path'], 
                  hyp_filename=file_path) * 100 
        print('\033[32;1mBLEU %s = %f\033[0m' % (corpus_info['evals'][k]['name'], bleu))
        summary = tf.Summary(value=[tf.Summary.Value(tag=('BLEU_%s'%corpus_info['evals'][k]['name']), simple_value=bleu)])
        summary_writer.add_summary(summary, i)
        summary_writer.flush()
    #break

                                             
print("END")
    

