import numpy as np
import matplotlib.pyplot as plt
import csv
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


if __name__ == '__main__':
    # incarcarea datelor de train si stabilirea dimensiunii unei imagini pentru a se potrivi arhitecturii folosite
    train_images = np.empty((13000, 227, 227, 3), dtype='float16')
    train_labels = []
    with open('train.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count_training = 0
        for row in csv_reader:
            if line_count_training == 0:
                line_count_training += 1
            else:
                train_image = row[0] # selectarea imaginii
                train_label = row[1] # selectarea etichetei
                image = Image.open('train_images/' + train_image) # incarcarea imaginii din folder-ul cu imagini de train
                image = tf.image.resize(image, (227, 227)) # redimensionarea imaginii pentru a se potrivi arhitecturii folosite
                #image = np.array(image) # transformarea imaginii intr-un array
                train_images[line_count_training - 1] = image # atasarea imaginii in array-ul cu imagini
                train_labels.append(int(train_label)) # atasarea label-ului liste de label-uri
                line_count_training += 1

    # incarcarea datelor de validation si stabilirea dimensiunii unei imagini pentru a se potrivi arhitecturii folosite
    validation_images = np.empty((1000, 227, 227, 3), dtype='float16')
    validation_labels = []
    with open('val.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                validation_image = row[0] # selectarea imaginii
                validation_label = row[1] # selectarea etichetei
                image = Image.open('val_images/' + validation_image) # incarcarea imaginii din folder-ul cu imagini de validation
                image = tf.image.resize(image, (227, 227)) # redimensionarea imaginii pentru a se potrivi arhitecturii folosite
                validation_images[line_count - 1] = image # atasarea imaginii in array-ul cu imagini
                validation_labels.append(int(validation_label)) # atasarea label-ului liste de label-uri
                train_images[line_count_training - 1] = image # atasarea imaginii in array-ul cu imagini
                train_labels.append(int(validation_label)) # atasarea label-ului liste de label-uri
                line_count += 1
                line_count_training += 1

    # incarcarea datelor de test si stabilirea dimensiunii unei imagini pentru a se potrivi arhitecturii folosite
    test_images = np.empty((5000, 227, 227, 3), dtype='float16')
    test_data = []
    with open('test.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                test_image = row[0] # selectarea imaginii
                test_data.append(test_image) # pastrarea denumirii imaginii intr-o lista care foloseste la afisarea in fisierul CSV
                image = Image.open('test_images/' + test_image) # incarcarea imaginii din folder-ul cu imagini de test
                image = tf.image.resize(image, (227, 227)) # redimensionarea imaginii pentru a se potrivi arhitecturii folosite
                test_images[line_count - 1] = image # atasarea imaginii in array-ul cu imagini
                line_count += 1


    # transformarea listelor in numpy arrays
    train_labels = np.array(train_labels)
    validation_labels = np.array(validation_labels)

    # construirea modelului
    CNN_model = keras.models.Sequential([
        keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu', padding='valid', input_shape=(227, 227, 3)),
        keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same'),
        keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'),
        keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'),
        keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        keras.layers.Flatten(), # aplatizarea datelor pentru a putea folosi layer-ul Dense
        keras.layers.Dense(2048, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(2048, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(96, activation='softmax') # 96 de neuroni reprezentand numarul de clase
    ])

    # compilarea modelului
    CNN_model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.SGD(learning_rate=0.01), metrics=['accuracy'])
    CNN_model.summary()

    # antrenarea modelului
    CNN_model.fit(train_images, train_labels, epochs=7, validation_data=(validation_images, validation_labels))

    # afisarea valorilor de loss si accuracy pentru validare
    print(CNN_model.evaluate(validation_images, validation_labels))

    # prezicerea etichetelor pentru datele de validare pentru a afla valorile pentru accuracy, precision, recall si pentru a construi matricea de confuzie
    predicted_validation_labels = CNN_model.predict(validation_images)
    predictions = []
    for prediction in predicted_validation_labels:
        p = prediction.argmax()
        predictions.append(p)
    predictions = np.array(predictions)

    # crearea matricii de confuzie pentru datele de validare
    confusion_matrix = confusion_matrix(validation_labels, predictions)
    plt.imshow(confusion_matrix, cmap='gray')
    plt.xlabel('Predicted labels', fontsize=10)
    plt.ylabel('Actual labels', fontsize=10)
    plt.title('Confusion Matrix for CNN', fontsize=10)
    plt.show()

    # afisarea valorii accuracy pentru datele de validare
    print("Accuracy score: ", accuracy_score(validation_labels, predictions))

    # afisarea valorii precision pentru datele de validare
    print("Precision score: ", precision_score(validation_labels, predictions, average='macro'))

    # afisarea valorii recall pentru datele de validare
    print("Recall score: ", recall_score(validation_labels, predictions, average='macro'))

    # clasificarea imaginilor de test
    predicted_test_labels = CNN_model.predict(test_images)

    g = open("submission.csv", "w")
    g.write("Image,Class\n")
    for i in range(len(test_data)):
        g.write(str(test_data[i]) + ',' + str(predicted_test_labels[i].argmax()) + '\n')
    g.close()