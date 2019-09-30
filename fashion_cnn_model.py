import tensorflow as tf 
import numpy as np 

#from Data_preparation import fashion_class_labels,load_fashion_data_feur_cnn,visualise

def build_fashion_cnn_model (input_images):
    input_layer = tf.reshape(input_images,[-1,28,28,1]) # input layer
    conv1 = tf.layers.conv2d(input_layer,filters=32,kernel_size=[5,5],padding="same",activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],strides=2) # pooling layer [-1,28,28,1] => [-1,14,14,1]
    convv2 = tf.layers.conv2d(pool1,filters=64,kernel_size=[5,5],padding="same",activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=convv2,pool_size=[2,2],strides=2) # 
    # pool2 auf eine Dimension reduziert
    pool2_flat = tf.reshape(pool2,[-1,7*7*64])
    #Dense layer bzw. Fully connected layer
    dense = tf.layers.dense(inputs=pool2_flat,units= 1024,activation=tf.nn.relu)
    #Dropout
    dropout = tf.layers.dropout(inputs=dense,rate=0.4)

    logits = tf.layers.dense (inputs=dropout,units = 10)
    return logits
