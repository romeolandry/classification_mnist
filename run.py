import tensorflow as tf 
from Data_preparation import *
import numpy as np 

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
input_picture = eval_data[0].reshape(28,28) # reshape in order to have image with size 28x28

input_label = fashion_class_labels[np.argmax(eval_labels[0],axis=None,out=None)]
visualise(input_picture,input_label)
