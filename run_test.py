import tensorflow as tf 
import numpy as np 

from Data_preparation import fashion_class_labels,visualise,load_fashion_data
from model import Model
# load Data 
train_data,train_labels,eval_data,eval_labels = load_fashion_data()
#visualisation
input_picture = eval_data[2].reshape(28,28) # reshape in order to have image with size 28x28
input_label = fashion_class_labels[np.argmax(eval_labels[2],axis=None,out=None)]
visualise(input_picture,input_label)

# load train model
paht_to_model = './model.ckpt.meta'
model = Model.predict(input_label,fashion_class_labels,paht_to_model)


