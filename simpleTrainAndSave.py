from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf

tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

data = tf.convert_to_tensor([
    [5.9,3.0,4.2,1.5, ],
    [6.9,3.1,5.4,2.1, ],
    [5.1,3.3,1.7,0.5, ],
    [6.0,3.4,4.5,1.6, ],
    [5.5,2.5,4.0,1.3, ],
    [6.2,2.9,4.3,1.3, ],
    [5.5,4.2,1.4,0.2, ],
    [6.3,2.8,5.1,1.5, ],
    [5.6,3.0,4.1,1.3, ],
    [6.7,2.5,5.8,1.8, ],
    [7.1,3.0,5.9,2.1, ],
    [4.3,3.0,1.1,0.1, ],
    [5.6,2.8,4.9,2.0, ],
    [5.5,2.3,4.0,1.3, ],
    [6.0,2.2,4.0,1.0, ],
    [5.1,3.5,1.4,0.2, ],
    [5.7,2.6,3.5,1.0, ],
    [4.8,3.4,1.9,0.2, ],
    [5.1,3.4,1.5,0.2, ],
    [5.7,2.5,5.0,2.0, ],
    [5.4,3.4,1.7,0.2, ],
    [5.6,3.0,4.5,1.5, ],
    [6.3,2.9,5.6,1.8, ],
    [6.3,2.5,4.9,1.5, ],
    [5.8,2.7,3.9,1.2, ],
    [6.1,3.0,4.6,1.4, ],
    [5.2,4.1,1.5,0.1, ],
    [6.7,3.1,4.7,1.5, ],
    [6.7,3.3,5.7,2.5, ],
    [6.4,2.9,4.3,1.3]
])

labels = tf.convert_to_tensor([
    [0.0, 1.0, 0.0, ],
    [0.0, 0.0, 1.0, ],
    [1.0, 0.0, 0.0, ],
    [0.0, 1.0, 0.0, ],
    [0.0, 1.0, 0.0, ],
    [0.0, 1.0, 0.0, ],
    [1.0, 0.0, 0.0, ],
    [0.0, 0.0, 1.0, ],
    [0.0, 1.0, 0.0, ],
    [0.0, 0.0, 1.0, ],
    [0.0, 0.0, 1.0, ],
    [1.0, 0.0, 0.0, ],
    [0.0, 0.0, 1.0, ],
    [0.0, 1.0, 0.0, ],
    [0.0, 1.0, 0.0, ],
    [1.0, 0.0, 0.0, ],
    [0.0, 1.0, 0.0, ],
    [1.0, 0.0, 0.0, ],
    [1.0, 0.0, 0.0, ],
    [0.0, 0.0, 1.0, ],
    [1.0, 0.0, 0.0, ],
    [0.0, 1.0, 0.0, ],
    [0.0, 0.0, 1.0, ],
    [0.0, 1.0, 0.0, ],
    [0.0, 1.0, 0.0, ],
    [0.0, 1.0, 0.0, ],
    [1.0, 0.0, 0.0, ],
    [0.0, 1.0, 0.0, ],
    [0.0, 0.0, 1.0, ],
    [0.0, 1.0, 0.0]
])

checkpoint_path = "C:/Users/USER/Downloads/training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1, period=10)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(3, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.GradientDescentOptimizer(0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

test_dataset = tf.convert_to_tensor([
    [5.1, 3.3, 1.7, 0.5, ],
    [5.9, 3.0, 4.2, 1.5, ],
    [6.9, 3.1, 5.4, 2.1]
])

test_labels = tf.convert_to_tensor([
    [1.0, 0.0, 0.0, ],
    [0.0, 1.0, 0.0, ],
    [0.0, 0.0, 1.0, ]
    ])

model.fit(data, labels, epochs=401, batch_size=10, callbacks=[cp_callback], validation_data = (test_dataset,test_labels))

print("Make some predictions:")
predict_dataset = tf.convert_to_tensor([
    [5.1, 3.3, 1.7, 0.5, ],
    [5.9, 3.0, 4.2, 1.5, ],
    [6.9, 3.1, 5.4, 2.1]
])



class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

# throws an error that the model is not compiled, so can be used for inference but not trained further
predictions = model(predict_dataset)

from functions import report

report(predictions, class_names)

