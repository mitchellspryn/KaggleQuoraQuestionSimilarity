import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk import wordnet
from nltk.stem import WordNetLemmatizer
import gensim
import logging
import re
import h5py
import random
import sys

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Input
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD, Adam
from keras.utils.io_utils import HDF5Matrix
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing import sequence
from keras.layers.merge import concatenate

from sklearn.model_selection import train_test_split

random.seed(42)

MAX_WORDS = 200000
MAX_SEQ_LEN = 30
EMBEDDING_DIM = 300
NUM_GRU = 175
NUM_DENSE = 100
TRAIN_PCT = 0.90

print('Initializing model...')
sys.stdout.flush()

print('Loading embedding model...')
sys.stdout.flush()
embedding_model = gensim.models.KeyedVectors.load_word2vec_format('../models/GoogleNews-vectors-negative300.bin', binary=True)

print('Reading data...')
sys.stdout.flush()
train_data = pd.read_json('../scratch/lstm_spaces_train.json')
test_data = pd.read_json('../scratch/lstm_spaces_test.json')

train_data = train_data.sample(frac=1) #shuffle

#Get all text and tokenize
print('Listifying...')
sys.stdout.flush()
train_1 = list(train_data['cleaned_question_1'])
train_2 = list(train_data['cleaned_question_2'])
test_1 = list(test_data['cleaned_question_1'])
test_2 = list(test_data['cleaned_question_2'])

train_labels = train_data['is_duplicate']
test_ids = test_data['test_id']

print('Tokenizing...')
sys.stdout.flush()
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(train_1 + train_2 + test_1 + test_2)

train_seq_1 = tokenizer.texts_to_sequences(train_1)
train_seq_2 = tokenizer.texts_to_sequences(train_2)
test_seq_1 = tokenizer.texts_to_sequences(test_1)
test_seq_2 = tokenizer.texts_to_sequences(test_2)

num_unique_tokens = len(tokenizer.word_index)
print('Found {0} unique tokens'.format(num_unique_tokens))

print('Padding sequences...')
sys.stdout.flush()
train_data_1 = sequence.pad_sequences(train_seq_1, maxlen=MAX_SEQ_LEN)
train_data_2 = sequence.pad_sequences(train_seq_2, maxlen=MAX_SEQ_LEN)
test_data_1 = sequence.pad_sequences(test_seq_1, maxlen=MAX_SEQ_LEN)
test_data_2 = sequence.pad_sequences(test_seq_2, maxlen=MAX_SEQ_LEN)

#Create embedding matrix
print('Creating embedding matrix...')
sys.stdout.flush()

with open('null_words_clean.txt', 'w') as f:
    embedding_matrix = np.zeros((min(MAX_WORDS, num_unique_tokens)+1, EMBEDDING_DIM))
    for word, i in tokenizer.word_index.items():
        if word in embedding_model.vocab:
            embedding_matrix[i] = embedding_model.word_vec(word)
        else:
            f.write('{0}\n'.format(word))
print('Null word embeddings: {0}'.format(np.sum(np.sum(embedding_matrix, axis=1) == 0)))

print('Dividing train/val data...')
sys.stdout.flush()
perm = np.random.permutation(len(train_data_1))
idx_train = perm[:int(len(train_data_1)*(TRAIN_PCT))]
idx_val = perm[int(len(train_data_1)*(TRAIN_PCT)):]

train_left_input = np.vstack((train_data_1[idx_train], train_data_2[idx_train]))
train_right_input = np.vstack((train_data_2[idx_train], train_data_1[idx_train]))
labels_train = np.concatenate((train_labels[idx_train], train_labels[idx_train]))

val_left_input = np.vstack((train_data_1[idx_val], train_data_2[idx_val]))
val_right_input = np.vstack((train_data_2[idx_val], train_data_1[idx_val]))
labels_val = np.concatenate((train_labels[idx_val], train_labels[idx_val]))

print('Building model...')
################################################################################
#
# MODEL DEFINITION START
#
################################################################################
embedding_layer = Embedding(min(MAX_WORDS, num_unique_tokens)+1,
        EMBEDDING_DIM,
        weights = [embedding_matrix],
        input_length = MAX_SEQ_LEN,
        trainable = False)

gru_forward_layer = LSTM(NUM_GRU, recurrent_dropout=0.1, dropout=0.1)
#gru_backward_layer = LSTM(NUM_GRU, recurrent_dropout=0.1, dropout=0.1, go_backwards=True)
#stack_batch_norm_layer = BatchNormalization()
#stack_dense_layer = Dense(128, activation='sigmoid')

left_input = Input(shape=(MAX_SEQ_LEN,), dtype='int32')
embedded_left = embedding_layer(left_input)
left_forward = gru_forward_layer(embedded_left)
#left_backward = gru_backward_layer(embedded_left)
#left_stacked = concatenate([left_forward, left_backward])
#left_out = stack_batch_norm_layer(left_stacked)
#left_out = stack_dense_layer(left_out)

right_input = Input(shape=(MAX_SEQ_LEN,), dtype='int32')
embedded_right = embedding_layer(right_input)
right_forward = gru_forward_layer(embedded_right)
#right_backward = gru_backward_layer(embedded_right)
#right_stacked = concatenate([right_forward, right_backward])
#right_out = stack_batch_norm_layer(right_stacked)
#right_out = stack_dense_layer(right_out)

merged = concatenate([left_forward, right_forward])
merged = Dropout(0.1)(merged)
merged = BatchNormalization()(merged)

merged = Dense(NUM_DENSE, activation='sigmoid')(merged)
merged = Dropout(0.1)(merged)
merged = BatchNormalization()(merged)

predictions = Dense(1, activation='sigmoid')(merged)

################################################################################
#
# MODEL DEFINITION END
#
################################################################################

model = Model([left_input, right_input], outputs=predictions)
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

model.summary()

model_name = 'two_column_maxword_{0}_seqlen_{1}_numgru_{2}_numdense_{3}'.format(MAX_WORDS, MAX_SEQ_LEN, NUM_GRU, NUM_DENSE)

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
best_model_path = model_name + '.h5'
model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, save_weights_only=True)

print('Beginning training...')
sys.stdout.flush()
hist = model.fit([train_left_input, train_right_input], labels_train,\
            validation_data=([val_left_input, val_right_input], labels_val),\
            epochs=200, batch_size=1024, shuffle=True,\
            class_weight=None, callbacks=[early_stopping, model_checkpoint])

print('Ending training. Loading best model...')
model.load_weights(best_model_path)
best_validation_score = min(hist.history['val_loss'])

print('Predicting...')
predictions = model.predict([test_data_1, test_data_2], batch_size=1024)
predictions = model.predict([test_data_2, test_data_1], batch_size=1024)
predictions /= 2

output = pd.DataFrame({'test_id':test_ids, 'is_duplicate': predictions.ravel()})
output = output[['test_id', 'is_duplicate']]
output.to_csv('../data/{0:.4f}_{1}.csv'.format(best_validation_score, model_name), index=False)
print('Done!')
