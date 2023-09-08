#!/usr/bin/env python3
import rospy
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import tf2_ros
import tf
from multi_object_tracker import Tracker

class PoseData():

    def __init__(self) -> None:
        self.subscriber = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.callback)

    def callback(self):
        pass


if __name__ == "__main__":
    rospy.init_node("product_tracker")
    
    while True:
        print("running")
        rospy.sleep(5)
