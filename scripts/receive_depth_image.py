#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# Visualize depth data from realsense camera

VISUALIZE_IMAGE = True



def callback(data):
    # When available, visualize depth data
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
    
    if VISUALIZE_IMAGE:
        cv2.imshow("depth image", cv_image)
        cv2.waitKey(1)


    

def main():
    rospy.init_node('depth_listener', anonymous=False)
    rospy.Subscriber("/camera/depth/image_rect_raw", Image, callback)
    rospy.spin()
    
if __name__ == "__main__":
    main()