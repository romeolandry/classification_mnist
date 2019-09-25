import tensorflow as tf 
import numpy as np

class model:
    def __init__ (self,eingabe_variable, ausgabe_variable,tuple_layer,dimension,number_of_classes):
        self.__eingabe_variable = eingabe_variable # X Tensor
        self.__ausgabe_variable = ausgabe_variable # Y tensor
        self.__tuple_layer = tuple_layer #  (256,126) für ein Model mit zwei schicht layer mit jeweils 256 und 126 layer
        self.__dimension = dimension # Input Layer des Models (28x28)
        self.__number_of_classes = number_of_classes # output des Models  bzw. Anzahl des Klasses, die vorherzusagen sind

    def fashion_model (self):
        
        # Definition of Layers
        first_weight_layer = tf.Variable(tf.random_normal([self.__dimension,self.__tuple_layer[0]],stddev=0.01),name="weight_1", dtype=tf.float32) # 256 is the number of neuron for the layer
        second_weight_layer = tf.Variable(tf.random_normal([self.__tuple_layer[0],self.__tuple_layer[1]],stddev=0.01), name="weight_2",dtype=tf.float32)
        last_weight_layer = tf.Variable(tf.random_normal([self.__tuple_layer[1],self.__number_of_classes],stddev=0.01),name="weight_last",dtype = tf.float32)

        # Bias für jede Schicht
        first_bias = tf.Variable(tf.random_normal([self.__tuple_layer[0]]), name= "bias", dtype = tf.float32)
        second_bias = tf.Variable(tf.random_normal([self.__tuple_layer[1]]), name="bias",dtype = tf.float32)
        last_bias = tf.Variable(tf.random_normal([self.__number_of_classes]), name ="bias",dtype = tf.float32)
        
        # Model construction 
        first_hidden_layer = tf.add(tf.matmul(self.__eingabe_variable,first_weight_layer),first_bias,name="first_hidden_layer")
        second_hiddene_layer = tf.add(tf.matmul(first_hidden_layer,second_weight_layer),second_bias, name="sacond_hidden_layer")
        output_layer = tf.add(tf.matmul(second_hiddene_layer,last_weight_layer),last_bias,name="output_layer")

        return output_layer

    def Loss (self,lernrate):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fashion_model(self.__eingabe_variable),labels=self.__ausgabe_variable))
        return (tf.train.AdamOptimizer(lernrate).minimize(cost), cost)

    def train (self,lernrate, X, Y, epoch,train_data,train_labels):   
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

    def predict (self,input_image,fashion_class_labels):
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

