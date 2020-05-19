import os
import sys
import tensorflow as tf
import numpy as np
import rospy
import cv2

from styx_msgs.msg import TrafficLight

# classifier inspired by https://github.com/likhtal/CarND-Capstone/ and https://gitlab.com/korvindeson/CarND-Capstone-master
# Since I accidentally erased my model, I am using frozen model from https://gitlab.com/korvindeson/CarND-Capstone-master/-/blob/master/ros/src/tl_detector/models/frozen_classifier_model.pb

NUM_CLASSES = 4
CATEGORY_INDEX = {1: {'id': 1, 'name': 'Green'}, 2: {'id': 2, 'name': 'Red'}, 3: {'id': 3, 'name': 'Yellow'}, 4: {'id': 4, 'name': 'off'}}

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        # this method is similar to https://github.com/likhtal/CarND-Capstone/blob/master/ros/src/tl_detector/light_classification/tl_classifier.py and https://github.com/vatsl/TrafficLight_Detection-TensorFlowAPI/blob/master/TrafficLightDetection-Inference.ipynb
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            graph_def = tf.GraphDef()
        
        with tf.gfile.GFile('./models/frozen_classifier_model.pb', 'rb') as fid:
            serialized_graph = fid.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def, name='')          

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="prefix")
            
        self.graph = graph
        
        self.loaded_x = self.graph.get_tensor_by_name('prefix/x:0')
        self.loaded_y = self.graph.get_tensor_by_name('prefix/y:0')
        self.loaded_keep_prob = self.graph.get_tensor_by_name('prefix/keep_prob:0')
        self.loaded_logits = self.graph.get_tensor_by_name('prefix/logits:0')
        self.loaded_acc = self.graph.get_tensor_by_name('prefix/accuracy:0')

    ### helper methods from : https://gitlab.com/korvindeson/CarND-Capstone-master/-/blob/master/ros/src/tl_detector/light_classification/tl_classifier.py
    
    def rgb2gray(self,x):
        return np.dot(x[...,:3], [0.299, 0.583, 0.114])


    def normalize(self,x):

        x_norm = []
        for img in x:
            x_norm.append(img/255)
        return x_norm

    def grayscale(self,x):
        x_gray = []
        for img in x:
            img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            x_gray.append(img2)
        return np.array(x_gray)


    def reshape(self,x):
        img2 = []
        for img in x:
            img_reshaped = img.reshape((32, 32, 1))
            img2.append(img_reshaped)
        return np.array(img2)

    def one_hot_encode(self,x, n_classes):

        all = []
    
        for label in x:

            lbl_vec = np.zeros(n_classes)
        
            lbl_vec[label] = 1.
        
            all.append(lbl_vec)
    

            return np.array(all)

    def preprocess(self,x):
        x = self.rgb2gray(x)
        x = self.normalize(x)
        x = self.reshape(x)
        return(x.astype(np.float32))
    
    def normalize_logits(self,logits):
        logits_norm = np.array(logits, copy=True) 


        for i in range(0, len(logits)):
            for j in range(0, len(logits[i])):
                logits_norm[i][j] = max(0, logits[i][j])
    
        for i in range(0, len(logits_norm)):
            sum = np.sum(logits_norm[i])
            logits_norm[i] = (logits_norm[i] / sum)

        return logits_norm


    ### helper methods done

    def get_classification(self, image_in):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        #TODO implement light color prediction
        # similar to https://gitlab.com/korvindeson/CarND-Capstone-master/-/blob/master/ros/src/tl_detector/light_classification/tl_classifier.py
        image = cv2.resize(image_in, (32, 32))
        batch = []
        batch.append(image)
        batch = np.asarray(batch)
        batch = self.preprocess(batch)
        with tf.Session(graph=self.graph) as sess:
            logits = sess.run(self.loaded_logits, feed_dict={self.loaded_x: batch, self.loaded_y: np.ndarray(shape=(1,3)), self.loaded_keep_prob: 1.0})
            logits = self.normalize_logits(logits)
            if ( (logits[0][0] > logits[0][1]) and (logits[0][0] > logits[0][2]) ):
                return TrafficLight.GREEN
            elif ( (logits[0][1] > logits[0][0]) and (logits[0][1] > logits[0][2]) ):
                return TrafficLight.YELLOW
            elif ( (logits[0][2] > logits[0][0]) and (logits[0][2] > logits[0][1]) ):
                return TrafficLight.RED

            return TrafficLight.UNKNOWN
