import tensorflow as tf 
from Data_preparation import fashion_class_labels,load_fashion_data,visualise
import numpy as np 
from model import Model

np.random.seed(20)
tf.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

# Initialisierung der tensoren Variable
dimension = 784 # 28x28 Pixels size of input image 
number_of_classes = 10 # number of classe to recognize by the Model

hidden_layer_1 = 256
hidden_layer_2 = 128

lernrate = 0.001
epochs = 20

## input 
X = tf.placeholder(tf.float32,[None,dimension],name="X")
##outpu 
Y = tf.placeholder(tf.float32,[None,number_of_classes],name="Y")
## weight of Model
w = tf.Variable(tf.random_normal([dimension,number_of_classes],stddev = 0.01),name="weights",dtype = tf.float32)
## Bias
#b = tf.Variable(tf.random_normal([number_of_classes]),name = "bias", dtype= tf.float32)

## large of Model as tuple  eg. (256,224) for a model with two hidden layer the first 256 neuron and the second 224 neuron
tuple_layer = (256,128)

# load Data 
train_data,train_labels,eval_data,eval_labels = load_fashion_data()

# Trainning of the Model
paht_to_model = './model_save/model.ckpt.meta' 
model = Model(tuple_layer,dimension,number_of_classes)
label = fashion_class_labels[np.argmax(eval_labels[2],axis=None,out=None)]
print(label)
#visualise(eval_data[2],label)
input_image = [eval_data[2]]
model.train(lernrate,epochs,train_data,train_labels,'./model_save/model.ckpt',input_image)
