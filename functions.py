from __future__ import absolute_import, division, print_function

import tensorflow as tf

def loss(m, x, y):
    y_ = m(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def report(predictions, class_names):
    for i, logits in enumerate(predictions):
        class_idx = tf.argmax(logits).numpy()
        p = tf.nn.softmax(logits)[class_idx]
        name = class_names[class_idx]
        print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100 * p))


def serializeModel(model):
    model.save('my_model.h5')
    print("model saved")


def deserializeModel():
    loaded_model = tf.keras.models.load_model('my_model.h5')
    print("model loaded")
    return loaded_model


