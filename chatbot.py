#buiding a chatbot
#libraries
import numpy as np
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import re
import time
#data
lines=open('movie_lines.txt',encoding='utf-8',errors='ignore').read().split('\n')
conversations=open('movie_conversations.txt',encoding='utf-8',errors='ignore').read().split('\n')
###preprocessing###
#lines dict
id2line={}
for line in lines:
    _line=line.split(' +++$+++ ')
    if len(_line)==5:
        id2line[_line[0]]=_line[4]
#conbersations dict
conversations_ids=[]
for conv in conversations[:-1]:
    _conv=conv.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    conversations_ids.append(_conv.split(','))
#qusetion and answer
qs=[]
ans=[]
for conv in conversations_ids:
    for i in range(len(conv)-1):
        qs.append(id2line[conv[i]])
        ans.append(id2line[conv[i+1]])
#text cleaning
def cleaner(text):
    text=text.lower()
    text=re.sub(r"i'm","i am",text)
    text=re.sub(r"'s"," is",text)
    text=re.sub(r"\'ll"," will",text)
    text=re.sub(r"\'ve"," have",text)
    text=re.sub(r"\'re"," are",text)
    text=re.sub(r"\'d","would",text)
    text=re.sub(r"won't","will not",text)
    text=re.sub(r"can't","can not",text)
    #text=re.sub(r"[--.,@#()*-+=&]","",text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text
#qs cleaning
clean_qs=[]
for q in qs:
    clean_qs.append(cleaner(q))
#ans cleaning
clean_ans=[]
for an in ans:
    clean_ans.append(cleaner(an))
#occurence dict
word2count={}
for q in clean_qs:
    for word in q.split():
        if word not in word2count:
            word2count[word]=1
        else:
            word2count[word]+=1
for an in clean_ans:
    for word in an.split():
        if word not in word2count:
            word2count[word]=1
        else:
            word2count[word]+=1
#question words to unique int
qs2int={}
threshold=20
word_number=0
for word , count in word2count.items():
    if word2count[word]>20:
        qs2int[word]=word_number
        word_number+=1
#answer words to unique int
ans2int={}
word_number=0
for word , count in word2count.items():
    if word2count[word]>20:
        ans2int[word]=word_number
        word_number+=1    
#some tokens
tokens=['<PAD>','<EOS>','<OUT>','<SOS>']
for token in tokens:
    qs2int[token]=len(qs2int)+1
    ans2int[token]=len(ans2int)+1
#reverse ans2int dict
int2ans={w_i:w for w, w_i in ans2int.items()}
#adding eos
for i in range(len(clean_ans)):
    clean_ans[i]+=" <EOS>"
#whole text into integers
questions_to_int=[]
for question in clean_qs:
    ints=[]
    for word in question:
        if word not in qs2int:
            ints.append(qs2int['<OUT>'])
        else:
            ints.append(qs2int[word])
    questions_to_int.append(ints)
answers_to_int=[]
for answer in clean_ans:
    ints=[]
    for word in answer:
        if word not in ans2int:
            ints.append(ans2int['<OUT>'])
        else:
            ints.append(ans2int[word])
    answers_to_int.append(ints)
#sort by qs length
sorted_clean_questions=[]
sorted_clean_answers=[]
for length in range(1,26):
    for i in enumerate(questions_to_int):
        if len(i[1])==length:
            sorted_clean_questions.append(questions_to_int[i[0]])
            sorted_clean_answers.append(answers_to_int[i[0]])
#ll=[]
#for j in enumerate(sorted_clean_questions):
#    ll.append(len(j[1]))


   
###model###
#placeholder creation
def model_inputs():
    inputs=tf.placeholder(tf.int32,[None,None],name='input')
    targets=tf.placeholder(tf.int32,[None,None],name='target')
    lr=tf.placeholder(tf.float32,name='learning_rate')
    keep_prob=tf.placeholder(tf.float32,name='keep_prob')
    return inputs,targets,lr,keep_prob
#target preprocess
def preprocess_targets(targets,word2int,batch_size):
    leftside=tf.fill([batch_size,1],word2int['<SOS>'])
    rightside=tf.strided_slice(targets,[0,0],[batch_size,-1],[1,1])
    preprocessed_target=tf.concat([leftside,rightside],axis=1)
    return preprocessed_target
#encoder layer
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                                                    cell_bw=encoder_cell,
                                                                    sequence_length=sequence_length,
                                                                    inputs=rnn_inputs,
                                                                    dtype=tf.float32)
    return encoder_state
#decode layer
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope,
                        output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states,
                                                                                                                                    attention_option="bahdanau",
                                                                                                                                    num_units=decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name="attn_dec_train")
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope=decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)    
# Decoding the test/validation set
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words,
                    decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states,
                                                                                                                                    attention_option="bahdanau",
                                                                                                                                    num_units=decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name="attn_dec_inf")
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope=decoding_scope)
    return test_predictions
# Creating the Decoder RNN
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope=decoding_scope,
                                                                      weights_initializer=weights,
                                                                      biases_initializer=biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
    return training_predictions, test_predictions
# Building the seq2seq model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words,
                  encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer=tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionswords2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions
###training the model###
#hyperparameters
epochs=50
batch_size=128
rnn_size=512
num_layers=3
encoding_embedding_size=512
decoding_embedding_size=512
learning_rate=0.01
learning_rate_dacay=0.9   #reduction of lr
min_learning_rate=0.0001
keep_probablity=0.5 #prob of hidden neorons to be present while training. keep in mind that this is not used for testing procedure.
#begin a session
tf.reset_default_graph()
session=tf.InteractiveSession()
#load model inputs
#tf.compat.v1.disable_eager_execution() 
inputs ,targets, lr, keep_prob= model_inputs()
#sequence length
sequence_length=tf.placeholder_with_default(25,None,name='sequence_length')
#shape inputs tensor
input_shape=tf.shape(inputs)
#training starts
training_predictions,test_predictions=seq2seq_model(tf.reverse(inputs,[-1]),
                                                    targets,
                                                    keep_prob,
                                                    batch_size,
                                                    sequence_length,
                                                    len(answers_to_int),
                                                    len(questions_to_int),
                                                    encoding_embedding_size,
                                                    decoding_embedding_size,
                                                    rnn_size,
                                                    num_layers,
                                                    questions_to_int)
#optimization and gradiant
with tf.name_scope('optimization'):
    loss_error=tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                targets,
                                                tf.ones([input_shape[0],sequence_length]))
    optimizer=tf.train.AdamOptimizer(learning_rate)
    gradients=optimizer.compute_gradients(loss_error)
    clipped_gradients=[(tf.clip_by_value(grad_tensor,-5.,5.),grad_variable)for grad_tensor , grad_variable in gradients if grad_tensor is not None]
    optimizer_gradient_clipping=optimizer.apply_gradients(clipped_gradients)
    
#padding the batches
def apply_padding(batch_of_sequences,word2int):
    max_seq_length=max([len(seq) for seq in batch_of_sequences])
    for seq in batch_of_sequences:
        if len(seq)< max_seq_length:
            seq=seq+[word2int['<PAD>']]*(max_seq_length-len(seq))
    return batch_of_sequences
#batch splitting qs and ans
def split_into_batches(questions,answers,batch_size):
    for batch_index in range(0,len(questions) // batch_size):
        start_index=batch_index*batch_size
        questions_in_batch=questions[start_index:start_index+batch_size]
        answers_in_batch=answers[start_index:start_index+batch_size]
        padded_questions_in_batch=np.array(apply_padding(questions_in_batch,questions_to_int))
        padded_answers_in_batch=np.array(apply_padding(answers_in_batch,answers_to_int))
        yield padded_questions_in_batch, padded_answers_in_batch
#train/validation splitting
split_index=int(len(sorted_clean_questions)*0.15)
training_questions=sorted_clean_questions[split_index:]
validation_questions=sorted_clean_questions[:split_index]
training_answers=sorted_clean_answers[split_index:]
validation_answers=sorted_clean_answers[:split_index]
# Training
batch_index_check_training_loss = 100
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 1000
checkpoint = "chatbot_weights.ckpt" # For Windows users, replace this line of code by: checkpoint = "./chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())
for epoch in range(1, epochs + 1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs: padded_questions_in_batch,
                                                                                               targets: padded_answers_in_batch,
                                                                                               lr: learning_rate,
                                                                                               sequence_length: padded_answers_in_batch.shape[1],
                                                                                               keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.format(epoch,
                                                                                                                                       epochs,
                                                                                                                                       batch_index,
                                                                                                                                       len(training_questions) // batch_size,
                                                                                                                                       total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                       int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch,
                                                                       targets: padded_answers_in_batch,
                                                                       lr: learning_rate,
                                                                       sequence_length: padded_answers_in_batch.shape[1],
                                                                       keep_prob: 1})
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('I speak better now!!')
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Sorry I do not speak better, I need to practice more.")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print("My apologies, I cannot speak better anymore. This is the best I can do.")
        break
print("Game Over")


        


























    
    
    
    