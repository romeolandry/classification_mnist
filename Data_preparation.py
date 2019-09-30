import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle
import matplotlib.pyplot as plt 
import numpy as np 

# class mapping
fashion_class_labels = {
    0:"T-shirt/shop",
    1:"Trouser",
    2:"Pullover",
    3:"Dress",
    4:"Coat",
    5:"Sandal",
    6:"Shirt",
    7:"Sneaker",
    8:"Bag",
    9:"Ankle boot"
}

def load_fashion_data ():
    data = input_data.read_data_sets('data/fashion',one_hot=True)

    #Traininfsdata
    train_data = data.train.images
    train_labels = data.train.labels

    #Evaluation Data
    eval_data = data.test.images
    eval_labels = data.test.labels

    # shuffle_Funtion zur Mischung und zum zufälligen Auswahl
    eval_data,eval_labels = shuffle(eval_data,eval_labels)
    train_data, train_labels = shuffle(train_data, train_labels)

    return (train_data,train_labels, eval_data, eval_labels)

def load_fashion_data_feur_cnn():
    """ Hier wird die Traindaten um eien 4D-Tensor umgewandeln [-1,28,28,1], mit 28X28X1 bzw. Breite höhe und Farbetiefe.
    die Farbtiefe wird umgesetzt, weil die Graustufen als farbekanal verwendet werden.
    und -1 steht für die initilsialisierung der Batchgröße.
    """
    train_data,train_labels,eval_data,eval_labels = load_fashion_data()
    train_data = train_data.reshape(-1,28,28,1)
    eval_data eval_data.reshape(-1,28,28,1)

    return (train_data,train_labels, eval_data, eval_labels)

def visualise (image, label):
    plt.title(label)
    plt.imshow(image.reshape(28,28))
    plt.show()