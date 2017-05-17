#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
    This is a modified version of the model developed by Bradley Pallen
    https://github.com/bradleypallen/keras-quora-question-pairs
'''

# import packages
import numpy as np
import csv, datetime, time
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, Reshape, Merge, BatchNormalization, TimeDistributed, Lambda
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils.data_utils import get_file
from keras import backend as K
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec


#%% Set the variables

DATA_DIR = '../Data/'
TRAINING_DATA_FILE = 'train.csv'
TEST_DATA_FILE = 'test.csv'
WORD_VECTOR = '/home/guillaume/Dropbox/DataScience/Data/GoogleNews-vectors-negative300.bin'
#QUESTION_PAIRS_FILE_URL = 'http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv'
#QUESTION_PAIRS_FILE = 'quora_duplicate_questions.tsv'
#GLOVE_ZIP_FILE_URL = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
#GLOVE_ZIP_FILE = 'glove.840B.300d.zip'
#GLOVE_FILE = 'glove.840B.300d.txt'
#Q1_TRAINING_DATA_FILE = 'q1_train.npy'
#Q2_TRAINING_DATA_FILE = 'q2_train.npy'
#LABEL_TRAINING_DATA_FILE = 'label_train.npy'
WORD_EMBEDDING_MATRIX_FILE = 'word_embedding_matrix.npy'
NB_WORDS_DATA_FILE = 'nb_words.json'
MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH = 25
EMBEDDING_DIM = 300
MODEL_WEIGHTS_FILE = 'question_pairs_weights.h5'
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1
RNG_SEED = 13371447
NB_EPOCHS = 25


#%% Load the data

question1 = []
question2 = []
is_duplicate = []

with open(DATA_DIR + TRAINING_DATA_FILE, encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter = ',')
    for row in reader:
        question1.append(row['question1'])
        question2.append(row['question2'])
        is_duplicate.append(row['is_duplicate'])

# At this point we have a list of all question1 and a list of all question2 + a
# list of labels
# We should have 404,290 pairs
print('Question pairs: %d' % len(question1))


#%% Build tokenized word index

questions = question1 + question2
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(questions)
question1_word_sequences = tokenizer.texts_to_sequences(question1)
question2_word_sequences = tokenizer.texts_to_sequences(question2)
word_index = tokenizer.word_index

print('Words in index: %d' % len(word_index))


#%% Word embedding

# we create an index of word vectors stored in a dictionary
# {word:vector}
embeddings_index = {}

#w2v_model = Word2Vec.load('w2v_300features_40minwords_20context.model')
import gensim
w2v_model = gensim.models.KeyedVectors.load_word2vec_format(WORD_VECTOR, binary=True)

for i in range(len(w2v_model.index2word)):
    word = w2v_model.index2word[i]
    embeddings_index[word] = w2v_model[word]

print('Word embedding: %d' % len(embeddings_index))


#%%

nb_words = min(MAX_NB_WORDS, len(embeddings_index))
word_embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > nb_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        word_embedding_matrix[i] = embedding_vector

print('Null word embeddings: %d' % np.sum(np.sum(word_embedding_matrix, axis=1) == 0))

#%% Generate tensors

q1_data = pad_sequences(question1_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
q2_data = pad_sequences(question2_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(is_duplicate, dtype='int')

print('Shape of question1 tensor: ', q1_data.shape)
print('Shape of question2 tensor: ', q2_data.shape)
print('Shape of label tensor: ', labels.shape)

# eventually save the data now


#%% Prepare the training and validation sets
X = np.stack((q1_data, q2_data), axis=1)
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT)

Q1_train = X_train[:,0]
Q2_train = X_train[:,1]
Q1_test = X_test[:,0]
Q2_test = X_test[:,1]


#%% Build the model

Q1 = Sequential()
Q1.add(Embedding(nb_words + 1, EMBEDDING_DIM, weights=[word_embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
Q1.add(TimeDistributed(Dense(EMBEDDING_DIM, activation='relu')))
Q1.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, )))
Q2 = Sequential()
Q2.add(Embedding(nb_words + 1, EMBEDDING_DIM, weights=[word_embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
Q2.add(TimeDistributed(Dense(EMBEDDING_DIM, activation='relu')))
Q2.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, )))
model = Sequential()
model.add(Merge([Q1, Q2], mode='concat'))
model.add(BatchNormalization())
model.add(Dense(200, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(200, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(200, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(200, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_acc', save_best_only=True)]


#%% Train the model

print('Starting training at ', datetime.datetime.now())

t0 = time.time()

history = model.fit([Q1_train, Q2_train],
                    y_train,
                    nb_epoch=NB_EPOCHS,
                    validation_split=VALIDATION_SPLIT,
                    verbose=1,
                    callbacks=callbacks)

t1 = time.time()

print('Training ended at ', datetime.datetime.now())
print('Minutes elapsed: %f' % ((t1 - t0) / 60.))

model.load_weights(MODEL_WEIGHTS_FILE)

accuracy = model.evaluate([Q1_test, Q2_test], y_test)
print('accuracy  = (0:.4f)'.format(accuracy))
'''loss, accuracy, precision, recall, fbeta_score = model.evaluate([Q1_test, Q2_test], y_test)
print('')
print('loss      = (0:.4f)'.format(loss))
print('accuracy  = (0:.4f)'.format(accuracy))
print('precision = (0:.4%)'.format(precision))
print('recall    = (0:.4%)'.format(recall))
print('F         = (0:,4%)'.format(fbeta_score))'''