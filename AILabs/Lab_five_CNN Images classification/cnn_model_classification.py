import tensorflow as tf
from tensorflow import keras

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
import math
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np


dataset_fm, md_fm = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
data_training, data_test = dataset_fm['train'], dataset_fm['test']

classes_type = md_fm.features['label'].names

generator_data = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
)

num_train_examples = md_fm.splits['train'].num_examples
num_test_examples = md_fm.splits['test'].num_examples

def processingData(images, names, dtype=tf.float32):
    images = tf.cast(images, dtype)
    images /= 255
    return images, names

data_training = data_training.map(processingData)
data_test = data_test.map(processingData)
data_training = data_training.cache()
data_test = data_test.cache()

def checkData(data_training, data_test):
    plt.figure(figsize=(10, 10))
    i = 0
    for (image, label) in data_test.take(25):
        image = image.numpy().reshape((28, 28))
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)
        plt.xlabel(classes_type[label])
        i += 1
    plt.show()

    print(f'\nNumber of training instances: {num_train_examples} pieces')
    print(f'Number of testing instances: {num_test_examples} pieces')


def checkImage(element, predictions_array, true_labels, images):
    predictions_array, true_label, img = predictions_array[element], true_labels[element], images[element]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img[..., 0], cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}%".format(classes_type[predicted_label], 100 * np.max(predictions_array), classes_type[true_label]), color=color)


def checkGraph(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('green')

def cnn_classification_learning(dataTrain, dataTest):
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
        tf.keras.layers.Conv2D(32, (2, 2), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (2, 2), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (2, 2), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(classes_type.__len__(), activation='softmax')
    ])

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    BATCH_SIZE = 32
    data_training = dataTrain.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
    data_test = dataTest.cache().batch(BATCH_SIZE)

    model.fit(data_training, epochs=10, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))

    test_loss, test_accuracy = model.evaluate(data_test, steps=math.ceil(num_test_examples / 32))
    print('\nAccuracy on test dataset:', test_accuracy)

    choice = input(f"\nDo you want to test a trained model?\n- Yes\n- No\n")

    if (choice.lower().__eq__("yes")):
        for test_images, test_labels in data_test.take(1):
            test_images = test_images.numpy()
            test_labels = test_labels.numpy()
            predictions = model.predict(test_images)
            predicted_class = np.argmax(predictions[0])
            class_name = classes_type[predicted_class]
            confidence = round(predictions[0][predicted_class] * 100, 2)
            print(f'\nPredicted class: {class_name}\nconfidence: {confidence}%')

            element = 0
            plt.figure(figsize=(6, 3))
            plt.subplot(1, 2, 1)
            checkImage(element, predictions, test_labels, test_images)
            plt.subplot(1, 2, 2)
            checkGraph(element, predictions, test_labels)
            plt.show()
    else:
        print("...")

while True:
    a = input("\nWhat do you want to choose?\n1) Check the correct data display.\n2) Train a machine learning model." +
              "\n\n\tEnter 'close', to exit.\n")

    match a:
        case "1":
            checkData(data_training, data_test)
        case "2":
            cnn_classification_learning(data_training, data_test)
        case "close":
            break
        case _:
            print("Try again.")
