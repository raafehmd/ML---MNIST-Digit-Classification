import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm.keras import TqdmCallback
import numpy as np
import matplotlib.pyplot as plt

dataset = keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = dataset.load_data()

X_train = keras.utils.normalize(X_train, axis=1)
X_test = keras.utils.normalize(X_test, axis=1)

model = keras.models.load_model('model.h5')

'''
# testing accuracy
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Tested Accuracy = ", test_acc * 100, "%")
'''

prediction = model.predict([X_test])

for i in range(45, 48):
    plt.grid(False)
    plt.imshow(X_test[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: {}".format(y_test[i]))
    plt.title("Prediction: {}".format(np.argmax(prediction[i])))
    plt.show()
