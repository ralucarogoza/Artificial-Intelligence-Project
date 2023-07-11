import numpy as np
import matplotlib.pyplot as plt
import csv
from PIL import Image
import copy
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

class KNN:
    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels
    def classify_image(self, test_image, number_of_neighbours=110, metric='l2'):
        if metric == 'l1': # implementare metrica L1 (distanta Manhattan)
            distances_neighbours = np.sum(np.abs(self.train_images - test_image), axis=1)
        elif metric == 'l2': # implementare metrica L2 (distanta euclidiana)
            distances_neighbours = np.sqrt(np.sum(((self.train_images - test_image) ** 2), axis=1))
        else:
            raise Exception("This metric wasn't implemented!")
        indices_neighbours = distances_neighbours.argsort() # ordonarea indicilor in functie de de distanta calculata
        indices_nearest_neighbours = indices_neighbours[:number_of_neighbours] # selectarea indicilor celor mai apropiati k vecini
        labels_nearest_neighbours = self.train_labels[indices_nearest_neighbours] # selectarea label-urilor celor mai apropiati k vecini
        return np.bincount(labels_nearest_neighbours).argmax() # returnarea label-ului predominant

    def classify_images(self, test_images, number_of_neighbours=110, metric='l2'):
        predicted_labels = []
        for image in test_images: # clasific fiecare imagine din datele de testare
            predicted_labels.append(self.classify_image(image, number_of_neighbours, metric))
        return np.array(predicted_labels) # transform lista intr-un numpy array


if __name__ == '__main__':
    # citirea datelor de train
    train_images = []
    train_labels = []
    with open('train.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                train_image = row[0] # selectarea imaginii
                train_label = row[1] # selectarea etichetei
                image = Image.open('train_images/' + train_image) # incarcarea imaginii din folder-ul cu imagini de train
                image_array = copy.deepcopy(np.asarray(image).flatten()) # transformarea imaginii intr-un array 1-dimensional
                train_images.append(image_array) # atasarea imaginii listei cu imagini
                train_labels.append(int(train_label)) # atasarea label-ului listei de label-uri
                line_count += 1

    # citirea datelor de validation
    validation_images = []
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
                image = Image.open('val_images/' + validation_image) # incarcarea imaginii din folder-ul cu imagini de validare
                image_array = copy.deepcopy(np.asarray(image).flatten()) # transformarea imaginii intr-un array 1-dimensional
                validation_images.append(image_array) # atasarea imaginii listei cu imagini
                validation_labels.append(int(validation_label)) # atasarea label-ului listei de label-uri
                train_images.append(image_array) # atasarea imaginii listei cu imagini
                train_labels.append(int(validation_label)) # atasarea label-ului listei de label-uri
                line_count += 1

    # citirea datelor de test
    test_images = []
    test_data = [] # denumirile imaginilor
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
                image_array = copy.deepcopy(np.asarray(image).flatten()) # transformarea imaginii intr-un array 1-dimensional
                test_images.append(image_array) # atasarea imaginii listei cu imagini
                line_count += 1

    # transformarea tuturor listelor in numpy array-uri
    train_images = np.array(train_images)
    validation_images = np.array(validation_images)
    test_images = np.array(test_images)
    train_labels = np.array(train_labels)
    validation_labels = np.array(validation_labels)

    KNN_model = KNN(train_images, train_labels)
    predicted_labels = KNN_model.classify_images(validation_images)

    # afisarea valorii accuracy pentru datele de validare
    print("Accuracy score: ", accuracy_score(validation_labels, predicted_labels))

    # afisarea valorii precision pentru datele de validare
    print("Precision score: ", precision_score(validation_labels, predicted_labels, average='macro'))

    # afisarea valorii recall pentru datele de validare
    print("Recall score: ", recall_score(validation_labels, predicted_labels, average='macro'))

    # crearea matricii de confuzie pentru datele de validare
    confusion_matrix = confusion_matrix(validation_labels, predicted_labels)
    print(confusion_matrix) # afisarea matricii de confuzie
    plt.imshow(confusion_matrix, cmap='gray')
    plt.xlabel('Predicted labels', fontsize=10)
    plt.ylabel('Actual labels', fontsize=10)
    plt.title('Confusion Matrix for KNN', fontsize=10)
    plt.show()

    predicted_labels = KNN_model.classify_images(test_images) # clasificarea imaginilor de test
    g = open("submission.csv", "w")
    g.write("Image,Class\n")
    for i in range(len(test_data)):
        g.write(str(test_data[i]) + ',' + str(predicted_labels[i]) + '\n')
    g.close()