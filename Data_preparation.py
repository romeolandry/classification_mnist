import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle
import matplotlib.pyplot as plt 


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

def visualisation (image, label):
    plt.title(label)
    plt.imshow(image,cmap='Greys')
    plt.show()