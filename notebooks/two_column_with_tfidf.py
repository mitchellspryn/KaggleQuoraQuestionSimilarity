########################################
## import packages
########################################
import os
import re
import csv
import codecs
import numpy as np
import pandas as pd
import h5py
import sys
import csv
import time

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

########################################
## set directories and parameters
########################################
BASE_DIR = '../data/'
TRAIN_DATA_FILE = BASE_DIR + 'augmented.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

TRAIN_PCT = 0.90

act = 'sigmoid'
re_weight = True # whether to re-weight classes to fit the 17.5% share in test set
#re_weight = False

if len(sys.argv) > 1:
    if (sys.argv[1] == '--wait'):
        print('Waiting for semaphore...')
        while not os.path.exists('../scratch/semaphore.txt'):
            time.sleep(1)            
        print('Found semaphore! Continuing...')

STAMP = 'two_column_with_tfidf'

print('Reading data...')
sys.stdout.flush()
data_1_train = np.load('../scratch/data_1_train.npy')
data_2_train = np.load('../scratch/data_2_train.npy')
labels_train = np.load('../scratch/labels_train.npy')
data_1_train_tfidf = np.load('../scratch/data_1_tfidf_train.npy')
data_2_train_tfidf = np.load('../scratch/data_2_tfidf_train.npy')

print('d1train shape: {0}'.format(data_1_train.shape))
print('d2train shape: {0}'.format(data_2_train.shape))
print('d1traintfidf shape: {0}'.format(data_1_train_tfidf.shape))
print('d2traintfidf shape: {0}'.format(data_2_train_tfidf.shape))

data_1_val = np.load('../scratch/data_1_val.npy')
data_2_val = np.load('../scratch/data_2_val.npy')
labels_val = np.load('../scratch/labels_val.npy')
data_1_val_tfidf = np.load('../scratch/data_1_val_tfidf.npy')
data_2_val_tfidf = np.load('../scratch/data_2_val_tfidf.npy')

train_data = pd.read_csv('../data/train_data_with_features.csv', index_col=False)
ids_val = np.load('../scratch/ids_val.npy')

test_data_1 = np.load('../scratch/test_data_1.npy')
test_data_2 = np.load('../scratch/test_data_2.npy')
test_ids = np.load('../scratch/test_ids.npy')
test_data_1_tfidf = np.load('../scratch/test_data_tfidf_1.npy')
test_data_2_tfidf = np.load('../scratch/test_data_tfidf_2.npy')

embedding_matrix = np.load('../scratch/embedding_matrix.npy')

#######################################
# define the model structure
#######################################
embedding_layer = Embedding(embedding_matrix.shape[0],
        embedding_matrix.shape[1],
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False)
lstm_words = LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)
lstm_tfidf = LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)
lstm_combined = LSTM(164, dropout=0.2, recurrent_dropout=0.2)

sequence_1_text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
sequence_1_tfidf_input = Input(shape=(MAX_SEQUENCE_LENGTH,1), dtype='float32')
embedded_sequences_1 = embedding_layer(sequence_1_text_input)
text_1_lstm = lstm_words(embedded_sequences_1)
#tfidf_1_lstm = lstm_tfidf(sequence_1_tfidf_input)
combined_1 = concatenate([text_1_lstm, sequence_1_tfidf_input])
lstm_1 = lstm_combined(combined_1)

sequence_2_text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
sequence_2_tfidf_input = Input(shape=(MAX_SEQUENCE_LENGTH,1), dtype='float32')
embedded_sequences_2 = embedding_layer(sequence_2_text_input)
text_2_lstm = lstm_words(embedded_sequences_2)
#tfidf_2_lstm = lstm_tfidf(sequence_2_tfidf_input)
combined_2 = concatenate([text_2_lstm, sequence_2_tfidf_input])
lstm_2 = lstm_combined(combined_2)

merged = concatenate([lstm_1, lstm_2])
merged = Dropout(0.2)(merged)

merged = Dense(256, activation='sigmoid')(merged)
merged = BatchNormalization()(merged)
merged = Dropout(0.2)(merged)

preds = Dense(1, activation='sigmoid')(merged)

########################################
## add class weight
########################################
weight_val = np.ones(len(labels_val))
class_weight = None
if re_weight:
    weight_val *= 0.472001959
    weight_val[labels_val==0] = 1.309028344
    class_weight = {0: 1.309028344, 1: 0.472001959}

########################################
## train the model
########################################
print('A')
sys.stdout.flush()
model = Model(inputs=[sequence_1_text_input, sequence_1_tfidf_input, sequence_2_text_input, sequence_2_tfidf_input], \
        outputs=preds)
print('B')
sys.stdout.flush()
model.compile(loss='binary_crossentropy',
        optimizer='nadam',
        metrics=['acc'])
print('C')
sys.stdout.flush()
model.summary()
print(STAMP)

early_stopping =EarlyStopping(monitor='val_loss', patience=3)
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

hist = model.fit([data_1_train, data_1_train_tfidf, data_2_train, data_2_train_tfidf], labels_train, \
        validation_data=([data_1_val, data_1_val_tfidf, data_2_val, data_2_val_tfidf], labels_val, weight_val), \
        epochs=200, batch_size=64, shuffle=True, \
        class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])

model.load_weights(bst_model_path)
bst_val_score = min(hist.history['val_loss'])

#########################################
## generate error file
#########################################
print('Making error file...')
data_1_val_pred = data_1_val[0:(data_1_val.shape[0]//2)]
data_2_val_pred = data_2_val[0:(data_2_val.shape[0]//2)]
data_1_val_tfidf_pred = data_1_val_tfidf[0:(data_1_val_tfidf.shape[0]//2)]
data_2_val_tfidf_pred = data_2_val_tfidf[0:(data_2_val_tfidf.shape[0]//2)]
ids_val_predict = ids_val[0:(ids_val.shape[0]//2)]

val_pred = model.predict([data_1_val_pred, data_1_val_tfidf_pred, data_2_val_pred, data_2_val_tfidf_pred], batch_size=32, verbose=1)

val_preds = pd.DataFrame(data = ids_val_predict, columns =['id'])
val_preds['predicted_prob'] = val_pred
val_preds['predicted_class'] = val_preds.apply(lambda r: 1 if r['predicted_prob'] > 0.5 else 0, axis=1)

joined_data = train_data.merge(val_preds, left_on='id', right_on='id', how='inner')

joined_data.to_csv('../scratch/errors_w.csv', index=False, quoting=csv.QUOTE_ALL)

########################################
## make the submission
########################################
print('Start making the submission before fine-tuning')

preds_fw = model.predict([test_data_1, test_data_1_tfidf, test_data_2, test_data_2_tfidf], batch_size=128, verbose=1)
preds_fwd = pd.DataFrame({'test_id':test_ids, 'is_duplicate': preds_fw.ravel()})
preds_bw = model.predict([test_data_2, test_data_2_tfidf, test_data_1, test_data_1_tfidf], batch_size=128, verbose=1)
preds_avg = (preds_fw + preds_bw) / 2.0
preds_avg = np.clip(preds_avg, 0.02, 0.98)

submission = pd.DataFrame({'test_id':test_ids, 'is_duplicate':preds_avg.ravel()})
submission.to_csv('tf_short_w_'%(bst_val_score)+STAMP+'.csv', index=False)
print('Done!')
