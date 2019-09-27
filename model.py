import tensorflow as tf 
import numpy as np
from Data_preparation import visualise,fashion_class_labels

class Model:

    def __init__ (self,tuple_layer,dimension,number_of_classes):
        self.__eingabe_variable = tf.placeholder(tf.float32,[None,dimension],name="X") # X Tensor
        self.__ausgabe_variable = tf.placeholder(tf.float32,[None,number_of_classes],name="Y") # Y tensor
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
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.fashion_model(),labels=self.__ausgabe_variable))
        train_op = tf.train.AdamOptimizer(lernrate).minimize(cost)
        return (train_op, cost)

    def train (self,lernrate,epochs,train_data,train_labels,path_to_save,input_image):
        train_op, cost = self.Loss(lernrate) 
        saver = tf.train.Saver() # hilft dazu, dass ein Model (Vairaible des Graphs) gespeichert wird
        model_fashion_out = self.fashion_model()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(epochs):
                sess.run(train_op, feed_dict={self.__eingabe_variable:train_data,self.__ausgabe_variable:train_labels})
                loss = sess.run(cost,feed_dict={self.__eingabe_variable:train_data,self.__ausgabe_variable:train_labels})
                accuracy = np.mean(np.argmax(sess.run(model_fashion_out,feed_dict = {self.__eingabe_variable:train_data,self.__ausgabe_variable:train_labels}),axis=1) == np.argmax(train_labels,axis=1))
                if (i%10 == 0):
                    print("Epoch {} // Accuracy: {} Loss: {}".format(i,accuracy,loss))
            print(" save....")
            saver.save(sess,path_to_save)
            print(" saved")

    @classmethod
    def predict (self,input_image,fashion_class_labels,path_model):
        with tf.Session() as sess:
            # Das Mdel wird jetzt 
            model_saver = tf.train.import_meta_graph(path_model) # aufladung des Graph Definition './model.ckpt.meta'
            model_saver.restore(sess, tf.train.latest_checkpoint('./'))
            # Initialisierung des geldene Graph als aktuele Graph
            current_graph = tf.get_default_graph()        
            # listet alle operationen auf, die beim Restauratieren des Graphen gespeichert wurde
            self.__eingabe_variable = current_graph.get_tensor_by_name("X:0")
            print("Tensor X: {} ".format(self.__eingabe_variable))
            restored_fashion_model = current_graph.get_tensor_by_name("output_layer:0") # restored_fashion_model beinhalted das Modell und damit kann das eval predict werden
            predictions = sess.run(restored_fashion_model,feed_dict = {self.__eingabe_variable:input_image})
            index = int(np.argmax(predictions,axis=1))
            # Vorhersage
            print("Gefundene Fashion-Kategorie : {}".format(fashion_class_labels[index]))
            #visualize
            visualise(input_image,fashion_class_labels[index])