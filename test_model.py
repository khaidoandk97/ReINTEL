# import libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub

import tensorflow_datasets as tfds
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews", 
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))

embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)
# hub_layer(train_examples_batch[:3])

model = keras.Sequential()
model.add(hub_layer)
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1))

model.compile(
    optimizer='adam',
    loss = keras.losses.BinaryCrossentropy(from_logits=True),
    metrics = ['accuracy']

)

history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=5,
                    validation_data=validation_data.batch(512),
                    verbose=1)