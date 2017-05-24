import os
import csv
import numpy as np
from keras.callbacks import ModelCheckpoint
from utils import to_categorical, get_comment_ids
from vdcnn import build_model


def get_input_data(file_path):
    raise NotImplementedError


def train(input_file, max_feature_length, num_classes, embedding_size, learning_rate, batch_size, num_epochs, save_dir=None, print_summary=False):
    # Stage 1: Convert raw texts into char-ids format && convert labels into one-hot vectors
    X_train, y_train = get_input_data(input_file)
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
