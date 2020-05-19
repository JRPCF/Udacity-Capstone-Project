import tensorflow as tf
import numpy as np
from PIL import Image
import os
import time
import tarfile
import rospy


# This file is from https://gitlab.com/korvindeson/CarND-Capstone-master/-/blob/master/ros/src/tl_detector/light_detect/tl_detect.py
# The models were also from there

MODELS_DIR=os.path.join(os.path.dirname(__file__),'tl_detect_models')
def load_graph(graph_file):
    """Loads a frozen inference graph"""
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        image_tensor = graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = graph.get_tensor_by_name('detection_scores:0')
        detection_classes = graph.get_tensor_by_name('detection_classes:0')
        
    return graph, image_tensor, detection_boxes,detection_scores,detection_classes

class TLDetect(object):
    def __init__(self):
        self.model_path= os.path.join(MODELS_DIR,'ssd_inception_v2_coco_2017_11_17.pb')
        
        self.detection_graph,self.image_tensor, self.detection_boxes,self.detection_scores,self.detection_classes = load_graph(self.model_path)
        rospy.loginfo("[TL Detect] -> Model loaded!")
        self.sess = tf.Session(graph=self.detection_graph) 
        rospy.loginfo("[TL Detect] -> Session created!")
        rospy.loginfo("[TL Detect] -> Check Camera_Sim checkbox")

    def to_image_coords(self,boxes, height, width):
        """
        The original box coordinate output is normalized, i.e [0, 1].
    
        This converts it back to the original coordinate based on the image
        size.
        """
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width
    
        return box_coords
        
    def filter_boxes(self,min_score, boxes, scores, classes, categories):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if classes[i] in categories and scores[i] >= min_score:
                idxs.append(i)
    
        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    
    def detect_boxes(self,
               image_tensor, 
               detection_boxes,
               detection_scores,
               detection_classes,
               image_np,
               runs=1):
        times = np.zeros(runs)
        for i in range(runs):
            t0 = time.time()
            (boxes, scores, classes) = self.sess.run([detection_boxes, detection_scores, detection_classes], 
                                                feed_dict={image_tensor: image_np})
            t1 = time.time()
            times[i] = (t1 - t0) * 1000
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)
        return boxes, scores, classes, times

        
    def detect_traffic_lights(self, image):
        width, height = image.size
        factor = 224.0 / width
        smaller_image = image.resize((int(width * factor), int(height * factor)), Image.ANTIALIAS)
        image_np = np.expand_dims(np.asarray(smaller_image, dtype=np.uint8), 0)
        boxes,scores,classes,times = self.detect_boxes(self.image_tensor,
                                      self.detection_boxes,
                                      self.detection_scores,
                                      self.detection_classes,
                                      image_np)
        confidence_cutoff = 0.2
        traffic_lights_class_id=10
        # Filter boxes with a confidence score less than `confidence_cutoff`
        boxes, scores,classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes,[traffic_lights_class_id])
        
        box_coords = self.to_image_coords(boxes, height, width)
        cropped_images = []
        
        for box in box_coords:
            top,left,bottom,right = box
            traffic_light = image.crop( (int(left), int(top), int(right), int(bottom)) )
            
            cropped_images.append(traffic_light)
        
        rospy.loginfo("[TLDetect] -> No. of Traffic light boxed: " + str(len(cropped_images)) 
                      + ", in " + str(np.sum(times)) + "ms")

        return cropped_images
