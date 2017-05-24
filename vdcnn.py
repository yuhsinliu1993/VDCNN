import os
import csv
import numpy as np
import tensorflow as tf
from Layers import ConvBlockLayer

from keras.models import Model
from keras.layers.convolutional import Conv1D
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense, Dropout, Lambda
from keras.layers.pooling import MaxPooling1D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

from utils import get_conv_shape, to_categorical, get_comment_ids


def get_input_data_from_csv(file_path, max_feature_length):
    comments = []
    classes = []
    sentiments = []
    with open(file_path) as f:
        reader = csv.DictReader(f, fieldnames=['id', 'comments', 'sentiment', 'class'])
        for i, row in enumerate(reader):
            if i > 0:
                comments.append(get_comment_ids(row['comments'], max_feature_length))
                sentiments.append(int(row['sentiment']) + 1)
                classes.append(int(row['class']))

    return np.asarray(comments, dtype='int32'), np.asarray(sentiments, dtype='int32'), np.asarray(classes, dtype='int32')


def build_model(num_filters, num_classes, sequence_max_length=512, num_quantized_chars=71, embedding_size=16, learning_rate=0.001, top_k=3, model_path=None):
    inputs = Input(shape=(sequence_max_length, ), dtype='int32', name='inputs')
    embedded_sent = Embedding(num_quantized_chars, embedding_size, input_length=sequence_max_length)(inputs)

    # First conv layer
    conv = Conv1D(filters=64, kernel_size=3, strides=2, padding="same")(embedded_sent)

    # Each ConvBlock with one MaxPooling Layer
    for i in range(len(num_filters)):
        conv = ConvBlockLayer(get_conv_shape(conv), num_filters[i])(conv)
        conv = MaxPooling1D(pool_size=3, strides=2, padding="same")(conv)

    # k-max pooling (Finds values and indices of the k largest entries for the last dimension)
    def _top_k(x):
        x = tf.transpose(x, [0, 2, 1])
        k_max = tf.nn.top_k(x, k=top_k)
        return tf.reshape(k_max[0], (-1, num_filters[-1] * top_k))
    k_max = Lambda(_top_k, output_shape=(num_filters[-1] * top_k,))(conv)

    # 3 fully-connected layer with dropout regularization
    fc1 = Dropout(0.2)(Dense(512, activation='relu', kernel_initializer='he_normal')(k_max))
    fc2 = Dropout(0.2)(Dense(512, activation='relu', kernel_initializer='he_normal')(fc1))
    fc3 = Dense(num_classes, activation='softmax')(fc2)

    # define optimizer
    sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=False)
    model = Model(inputs=inputs, outputs=fc3)
    model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])

    if model_path is not None:
        model.load_weights(model_path)

    return model


def train(input_file, max_feature_length, num_classes, embedding_size, learning_rate, batch_size, num_epochs, save_dir=None, print_summary=False):
    # Stage 1: Convert raw texts into char-ids format && convert labels into one-hot vectors
    X_train, y_train = get_input_data_from_csv(input_file, max_feature_length)
    y_train = to_categorical(y_train, num_classes)

    # Stage 2: Build Model
    num_filters = [64, 128, 256, 512]

    model = build_model(num_filters=num_filters, num_classes=num_classes, embedding_size=embedding_size, learning_rate=learning_rate)

    # Stage 3: Training
    save_dir = save_dir if save_dir is not None else 'checkpoints'
    filepath = os.path.join(save_dir, "weights-{epoch:02d}-{val_acc:.2f}.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    if print_summary:
        print(model.summary())

    model.fit(
        x=X_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.33,
        callbacks=[checkpoint],
        shuffle=True,
        verbose=True
    )
