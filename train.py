import tensorflow as tf
from preprocess import preprocess_image, char_to_num, num_to_char
import pandas as pd
import numpy as np
import os

max_label_len = 6  # e.g., "005376"
num_classes = len(char_to_num) + 1  # +1 for blank

def encode_label(label):
    return [char_to_num[c] for c in label]

def load_data(label_path='labels.csv', img_dir='images/'):
    df = pd.read_csv(label_path)
    X, y = [], []
    for _, row in df.iterrows():
        img = preprocess_image(os.path.join(img_dir, row['filename']))
        X.append(img)
        y.append(encode_label(str(row['label']).zfill(max_label_len)))
    return np.array(X), y

def create_model(input_shape=(32, 128, 1)):
    inputs = tf.keras.layers.Input(shape=input_shape, name='input')
    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Reshape((32, -1))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs, x)
    return model

def ctc_loss(y_true, y_pred):
    input_len = tf.ones(shape=(len(y_pred), 1)) * y_pred.shape[1]
    label_len = tf.ones(shape=(len(y_true), 1)) * max_label_len
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_len, label_len)

def train():
    X, raw_labels = load_data()
    y = tf.keras.preprocessing.sequence.pad_sequences(raw_labels, maxlen=max_label_len, padding='post', value=num_classes - 1)
    y = np.array(y)

    model = create_model()
    model.compile(optimizer='adam', loss=ctc_loss)
    model.fit(X, y, batch_size=1, epochs=50) #, validation_split=0.1)
    model.save("model.h5")

if __name__ == '__main__':
    train()
