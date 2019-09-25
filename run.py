import tensorflow as tf 
from Data_preparation import *
import numpy as np 
from model import *

np.random.seed(20)
tf.logging.set_verbosity(tf.logging.DEBUG)
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

# load Data 
train_data,train_labels,eval_data,eval_labels = load_fashion_data()
#visualisation
input_picture = eval_data[2].reshape(28,28) # reshape in order to have image with size 28x28

input_label = fashion_class_labels[np.argmax(eval_labels[2],axis=None,out=None)]
#visualise(input_picture,input_label)

# Initialisierung der tensoren Variable
dimension = 784 # 28x28 Pixels
number_of_classes = 10

hidden_layer_1 = 256
hidden_layer_2 = 128

## input 
X = tf.placeholder(tf.float32,[None,dimension],name="X")
##outpu 
Y = tf.placeholder(tf.float32,[None,number_of_classes],name="Y")
## weight of Model
w = tf.Variable(tf.random_normal([dimension,number_of_classes],stddev = 0.01),name="weights",dtype = tf.float32)
## Bias
b = tf.Variable(tf.random_normal([number_of_classes]),name = "bias", dtype= tf.float32)

# Trainning of the Model
