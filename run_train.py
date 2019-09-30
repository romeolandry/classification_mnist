import tensorflow as tf 
import numpy as np 
from model import Model

from Data_preparation import fashion_class_labels,load_fashion_data,visualise, load_fashion_data_feur_cnn, get_next_batch
from fashion_cnn_model import build_fashion_cnn_model

np.random.seed(20)
tf.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

def train_mlp ():

    # Initialisierung der tensoren Variable
    dimension = 784 # 28x28 Pixels size of input image 
    number_of_classes = 10 # number of classe to recognize by the Model
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
    paht_to_model = './model_save/model.ckpt' 
    model = Model(tuple_layer,dimension,number_of_classes)
    label = fashion_class_labels[np.argmax(eval_labels[2],axis=None,out=None)]
    print(label)
    #visualise(eval_data[2],label)
    input_image = [eval_data[2]]
    model.train(lernrate,epochs,train_data,train_labels,paht_to_model,input_image)

def train_cnn(path_to_save):
    train_data,train_labels, eval_data,eval_labels = load_fashion_data_feur_cnn()

    # placehalter  fuer Bildern und labeln
    
    images = tf.placeholder("float",[None,28,28,1], "images")
    labels = tf.placeholder("float",[None,10], "labels")

    pred = build_fashion_cnn_model(images)
    correct_prediction_op = tf.equal(tf.argmax(pred,1), tf.argmax(labels,1))

    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction_op,tf.float32))

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits= pred,labels=labels))
    train_op = tf.train.AdamOptimizer(0.001).minimize(loss_op)

    EPOCH_NUM = 100
    epochs = range(0,EPOCH_NUM)
    BATCH_SIZE = 64

    batches = range(0,len(train_labels) // BATCH_SIZE)
    saver = tf.train.Saver() # hilft dazu, dass ein Model (Vairaible des Graphs) gespeichert wird

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in epochs:
            for b in batches:
                img_input = get_next_batch(train_data,b,BATCH_SIZE) # 
                labels_output = get_next_batch(train_labels,b,BATCH_SIZE)#
                sess.run(train_op,feed_dict = {images:img_input,labels:labels_output}) #
                loss,acc = sess.run([loss_op,accuracy_op],feed_dict = {images:img_input,labels:labels_output})

            print("Epoch: {} Accuracy: {} Loss: {}".format(i,acc,loss))
        print(" save....")
        saver.save(sess,path_to_save)
        print(" saved")
        
print("train cnn model")
train_cnn("./model_cnn_save/model.ckpt")