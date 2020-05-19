#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from scipy.spatial import KDTree
from light_classification.tl_classifier import TLClassifier
# Light detector from https://gitlab.com/korvindeson/CarND-Capstone-master
from light_detect.tl_detect import TLDetect
from scipy import stats
import tf
import cv2
import yaml
import PIL
import numpy as np
from styx_msgs.msg import TrafficLight

# My detector is inspired by https://github.com/likhtal/CarND-Capstone who used https://becominghuman.ai/traffic-light-detection-tensorflow-api-c75fdbadac62
# Detector imports

#

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        """
        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        """
        self.lights_prev = None
        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        
        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.light_detector = TLDetect()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
       
        
        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint
        Args:
            msg (Image): image from car-mounted camera
        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to
        Returns:
            int: index of the closest waypoint in self.waypoints
        """
        #TODO implement
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        return closest_idx

    def get_light_state(self, light):
        #return light.state
        """Determines the current color of the traffic light
        Args:
            light (TrafficLight): light to classify
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
    
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        image2PIL = PIL.Image.fromarray(cv_image)
        #Get classification
        lights_boxed = self.light_detector.detect_traffic_lights(image2PIL)
        if len(lights_boxed) > 0:
            lights_all = []
            for i in range(len(lights_boxed)):
                lights_all.append(self.light_classifier.get_classification(np.array(lights_boxed[0])))
            lights_all = np.array(lights_all)
            lights_state = stats.mode(lights_all)
            #self.light_classifier.get_classification(np.array(lights_boxed[0]))
        else:
            lights_state = None

        if(not lights_state or not (lights_state==light.state)):
            rospy.loginfo('failed')
        else:
            rospy.loginfo('success')
            
        #return light.state
        rospy.loginfo(lights_state)
        if lights_state is not None and (self.lights_prev == TrafficLight.RED or self.lights_prev == TrafficLight.YELLOW):
            self.lights_prev = lights_state
            lights_state = TrafficLight.RED
        else:
            self.lights_prev = lights_state
        if lights_state is not None:
            return lights_state
        rospy.loginfo('failure')
        return 0
   

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color
        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        closest_light = None
        line_wp_idx = None
        
        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

        #TODO find the closest visible traffic light (if one exists)
        diff = len(self.waypoints.waypoints)
        for i, light in enumerate(self.lights):
            line = stop_line_positions[i]
            temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
            d = temp_wp_idx - car_wp_idx
            if d >= 0 and d < diff:
                diff = d
                closest_light = light
                line_wp_idx = temp_wp_idx

        if closest_light:
            state = self.get_light_state(closest_light)
            return line_wp_idx, state
         
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')