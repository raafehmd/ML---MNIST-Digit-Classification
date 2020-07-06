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

'''
# Printing raw image data
print(X_train[0])

# Printing actual image
plt.imshow(X_train[0], cmap=plt.cm.binary)
plt.show()
'''

# model
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(56, activation='relu'),
    layers.Dense(28, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# training model
model.fit(X_train, y_train, epochs=10, verbose=0, callbacks=[TqdmCallback(verbose=0)])

model.save('model.h5')

# testing accuracy
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Tested Accuracy = ", test_acc * 100, "%")

prediction = model.predict([X_test])

for i in range(3):
    plt.grid(False)
    plt.imshow(X_test(i), cmap=plt.cm.binary)
    plt.xlabel("Actual: {}".format(y_test[i]))
    plt.title("Prediction: {}".format(np.argmax(prediction[i])))
