import tensorflow as tf 
import numpy as np

def fashion_model (X):
    dimension = 784 # 28x28 Pixels
    number_of_classes = 10

    hidden_layer_1 = 256
    hidden_layer_2 = 128

    # Definition of Layers 
    first_weight_layer = tf.Variable(tf.random_normal([dimension,hidden_layer_1],stddev=0.01),name="weight_1", dtype=tf.float32) # 256 is the number of neuron for the layer
    second_weight_layer = tf.Variable(tf.random_normal([hidden_layer_1,hidden_layer_2],stddev=0.01), name="weight_2",dtype=tf.float32)
    last_weight_layer = tf.Variable(tf.random_normal([hidden_layer_2,number_of_classes],stddev=0.01),name="weight_last",dtype = tf.float32)

    # Bias für jede Schicht
    first_bias = tf.Variable(tf.random_normal([hidden_layer_1]), name= "bias", dtype = tf.float32)
    second_bias = tf.Variable(tf.random_normal([hidden_layer_2]), name="bias",dtype = tf.float32)
    last_bias = tf.Variable(tf.random_normal([number_of_classes]), name ="bias",dtype = tf.float32)
    
    # Model construction 
    first_hidden_layer = tf.add(tf.matmul(X,first_weight_layer),first_bias,name="first_hidden_layer")
    second_hiddene_layer = tf.add(tf.matmul(first_hidden_layer,second_weight_layer),second_bias, name="sacond_hidden_layer")
    output_layer = tf.add(tf.matmul(second_hiddene_layer,last_weight_layer),last_bias,name="output_layer")

    return output_layer

def Loss (lernrate,X,Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fashion_model(X),labels=Y))
    return (tf.train.AdamOptimizer(lernrate).minimize(cost), cost)

def train (lernrate, X, Y, epoch,train_data,train_labels):   
    saver = tf.train.Saver() # hilft dazu, dass ein Model (Vairaible des Graphs) gespeichert wird
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in epoch:
            sess.run(Loss(lernrate,X,Y)[0], feed_dict={X:train_data,Y:train_labels})
            loss = sess.run(Loss(lernrate,X,Y)[1],feed_dict={X:train_data,Y:train_labels})
            accuracy = np.mean(np.argmax(sess.run(fashion_model,feed_dict = {X:train_data,Y:train_labels}),axis=1) == np.argmax(train_labels,axis=1))
            if (i%10 == 0):
                print("Epoch {} // Accuracy: {} Loss: {}".format(i,accuracy,loss))
        saver.save(sess,"./model.ckpt")

def predict (input_image,fashion_class_labels):
    with tf.Session() as sess:
        # Das Mdel wird jetzt 
        model_saver = tf.train.import_meta_graph('./model.ckpt.meta') # aufladung des Graph Definition
        model_saver.restore(sess, tf.train.latest_checkpoint('./'))
        # Initialisierung des geldene Graph als aktuele Graph
        current_graph = tf.get_default_graph()        
        # listet alle operationen auf, die beim Restauratieren des Graphen gespeichert wurde
        print(current_graph.get_operation_by_name())
        # zugriff auf den Tensor X
        X = current_graph.get_tensor_by_name("X:0")
        print("Tensor X: ".format(X))
        restored_fashion_model = current_graph.get_tensor_by_name("fashion_model:0") # restored_fashion_model beinhalted das Modell und damit kann das eval predict werden

        predictions = sess.run(restored_fashion_model,feed_dict = {X:input_image})
        index = int(np.argmax(predictions,axis=1))
        # Vorhersage
        print("Gefundene Fashion-Kategorie : {}".format(fashion_class_labels[index]))

