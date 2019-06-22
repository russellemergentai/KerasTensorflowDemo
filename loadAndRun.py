import tensorflow as tf

tf.enable_eager_execution()

from functions import deserializeModel
model = deserializeModel()

# try an individual test
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

