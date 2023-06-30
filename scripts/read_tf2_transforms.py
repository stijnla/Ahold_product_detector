#!/usr/bin/env python  

import rospy
import tf2_ros
import geometry_msgs.msg
from tf.transformations import euler_from_quaternion
from std_msgs.msg import Int32
import numpy as np
from ahold_product_detection.msg import ProductList, ProductClass


class DetectedProductReader():
    
    def __init__(self) -> None:
        rospy.init_node("tf2_listener")
        self.pub = rospy.Publisher("detected_products", ProductList, queue_size=1)
        self.sub = rospy.Subscriber("detected_classes", ProductClass, self.product_class_callback)

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.max_num_objects = 0
        
        self.detected_object_classes = []


    def get_transform_values(self,t):
        x = t.transform.translation.x
        y = t.transform.translation.y
        z = t.transform.translation.z
        q = [0, 0, 0, 0]
        q[0] = t.transform.rotation.x
        q[1] = t.transform.rotation.y
        q[2] = t.transform.rotation.z
        q[3] = t.transform.rotation.w
        euler = euler_from_quaternion(q)

        pose = np.array([0, 0, x, y, z, euler[0], euler[1], euler[2]])
        return pose



    def product_class_callback(self, message):
        stamp = message.header.stamp
        object_class = message.classification
        object_score = message.score
        self.detected_object_classes.append((stamp, object_class, object_score))
        


    def read(self):
        while not rospy.is_shutdown():
            
            detected_object_poses = []
            products = ProductList()

            for i in range(self.max_num_objects):

                try:
                    if False: # robot
                        t = self.tfBuffer.lookup_transform('base_link', 'object'+str(i), rospy.Time(0))

                    else:
                        t = self.tfBuffer.lookup_transform('camera_color_optical_frame', 'object'+str(i), rospy.Time(0))
                    
                    
                    try:
                        stamp, object_class, object_score = self.detected_object_classes[i]
                        rospy.loginfo(object_class)
                        rospy.loginfo(str(stamp.nsecs) + str(" ~ ") + str(t.header.stamp.nsecs))
                    except:
                        # Detection is a filler
                        object_class = None
                        object_score = None
                        stamp = None
                    
                    pose = self.get_transform_values(t)
                    
                    # check if pose is of a real object of a filler (pose with all zeros)
                    if np.any(pose):
                        pose[0] = object_class
                        pose[1] = object_score
                        if pose[0] == None:
                            rospy.logerr("CLASS IS NONE")
                        detected_object_poses.append(pose)
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                    rospy.logerr("Num objects is too large, this should not occur " + str(self.max_num_objects))

            self.detected_object_classes = []

            products.data = detected_object_poses
            self.pub.publish(products)
            
            # Update maximum detected products
            message = rospy.wait_for_message("number_of_objects", Int32)
            self.max_num_objects = message.data



def main():
    d = DetectedProductReader()
    d.read()
    


if __name__ == "__main__":
    main()