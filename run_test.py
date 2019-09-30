import tensorflow as tf 
import numpy as np 

from Data_preparation import fashion_class_labels,visualise,load_fashion_data,load_fashion_data_feur_cnn
from model import Model

def test_mlp ():
    # load Data 
    train_data,train_labels,eval_data,eval_labels = load_fashion_data()
    #visualisation
    input_picture = eval_data[2] # reshape in order to have image with size 28x28
    input_label = fashion_class_labels[np.argmax(eval_labels[2],axis=None,out=None)]
    visualise(input_picture,input_label)

    # load train model
    paht_to_model = './model_save/model.ckpt.meta' # classification_mnist/model_save/model.ckpt.meta
    input_picture = [eval_data[2]]
    model = Model.predict(input_picture,fashion_class_labels,paht_to_model)

def test_cnn_model(paht_to_model_meta):
    train_data,train_labels, eval_data,eval_labels = load_fashion_data_feur_cnn()

    with tf.Session() as sess:
        # Das Mdel wird jetzt 
        model_saver = tf.train.import_meta_graph(paht_to_model_meta) # aufladung des Graph Definition './model.ckpt.meta'
        model_saver.restore(sess, tf.train.latest_checkpoint('./'))
        # Initialisierung des geldene Graph als aktuele Graph
        current_graph = tf.get_default_graph()        
        # listet alle operationen auf, die beim Restauratieren des Graphen gespeichert wurde
        images = current_graph.get_tensor_by_name("images")
        print("Tensor images: {} ".format(images))
        restored_fashion_model = current_graph.get_tensor_by_name("logits:0") # restored_fashion_model beinhalted das Modell und damit kann das eval predict werden
        predictions = sess.run(restored_fashion_model,feed_dict = {images:eval_data[0]})
        index = int(np.argmax(predictions,axis=1))
        # Vorhersage
        print("Gefundene Fashion-Kategorie : {}".format(fashion_class_labels[index]))
        #visualize
        #visualise(input_image,fashion_class_labels[index])