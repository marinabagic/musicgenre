import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer

# Fields
def load_data_manually(path):
    sentences = []
    labels = []
    label_to_index = {'Metal': 0,
                      'Pop': 1,
                      'Jazz': 2,
                      'Rock': 3,
                      'Folk': 4}
    with open(path, encoding='utf-8', mode='r') as in_file:
        for line in in_file:
            vals = line.strip().split(',')
            sentences.append(vals[4].split(' '))
            labels.append(label_to_index[vals[5]])
    return sentences, labels


train_s, train_l = load_data_manually('drive/MyDrive/Genre_Classification/train_eng/train_english.csv')
valid_s, valid_l = load_data_manually('drive/MyDrive/Genre_Classification/valid_eng/valid_english.csv')
test_s, test_l = load_data_manually('drive/MyDrive/Genre_Classification/test_eng/test_english.csv')
num_words = 10000  # number of unique words
tokenizer = Tokenizer(num_words, lower=True)
df_total = train_s + valid_s + test_s
tokenizer.fit_on_texts(df_total)

from keras.preprocessing.sequence import pad_sequences

X_train = tokenizer.texts_to_sequences(train_s)
X_train_pad = pad_sequences(X_train,maxlen=300, padding='post')
X_test = tokenizer.texts_to_sequences(test_s)
X_test_pad = pad_sequences(X_test, maxlen=300, padding='post')
X_val = tokenizer.texts_to_sequences(valid_s)
X_val_pad = pad_sequences(X_val, maxlen=300, padding='post')

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(train_l)
y_test = to_categorical(test_l)
y_val = to_categorical(valid_l)

import gensim.downloader as api
glove_gensim  = api.load('word2vec-google-news-300') #300 dimension

vector_size = 300
gensim_weight_matrix = np.zeros((num_words ,vector_size))

counter = 0
for word, index in tokenizer.word_index.items():
    if index < num_words: # since index starts with zero
        if word in glove_gensim.wv.vocab:
            gensim_weight_matrix[index] = glove_gensim[word]
            counter += 1
        else:
            gensim_weight_matrix[index] = np.zeros(300)

print('Embedano ' + str(counter) + ' rijeci!')


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Embedding,Bidirectional, Reshape
from tensorflow.keras.layers import Dropout
from tensorflow.keras import optimizers


adam = optimizers.Adam(learning_rate=0.0001)
EMBEDDING_DIM = 300
class_num = 5
model = Sequential()
model.add(Embedding(input_dim=num_words,
    output_dim=EMBEDDING_DIM,
    input_length=X_train_pad.shape[1],
    weights=[gensim_weight_matrix], trainable=False))
model.add(Dropout(0.5))
model.add(SimpleRNN(256, return_sequences=True))
model.add(Dropout(0.5))
model.add(SimpleRNN(128, return_sequences=True))
model.add(Dropout(0.5))
model.add(SimpleRNN(256, return_sequences=False))
model.add(Dense(class_num, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics='accuracy')

#EarlyStopping and ModelCheckpoint
from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=8)
mc = ModelCheckpoint('./model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)


history_embedding = model.fit(X_train_pad, y_train,
                                epochs=25, batch_size=128,
                                validation_data=(X_val_pad, y_val),
                                verbose=1, callbacks=[es, mc])


y_pred = np.argmax(model.predict(X_test_pad), axis=1)
y_true = np.argmax(y_test, axis=1)
from sklearn import metrics
print(metrics.classification_report(y_pred, y_true, labels=[0, 1, 2, 3, 4], digits=4))
print(metrics.confusion_matrix(y_true, y_pred))
