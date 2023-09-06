#!/usr/bin/env python3

import rospy
from cv_bridge import CvBridge
from ahold_product_detection.srv import *
import cv2
import numpy as np
from pynput.keyboard import Key, Listener, KeyCode
import time



def requestRGBD_client():
    """Returns message received from rgbd_processor"""

    # Get RGBD data when ready from the rgbd_processor node
    rospy.wait_for_service('request_rgbd')
    
    try:
        request_rgbd = rospy.ServiceProxy('request_rgbd', RequestRGBD)
        response = request_rgbd(0)
        
        if response.data_available:
            return response
        else:
            return None
    
    except rospy.ServiceException as e:
        rospy.logwarn("Service call failed: %s"%e)



def read_message(message):
    """Returns rgb image, depth image, camera intrinsics, and the corresponding frame id"""
    image_msg, pointcloud_msg, camera_info_msg = message.images, message.pointcloud, message.intrinsics
    frame_id = pointcloud_msg.header.frame_id
    
    # Convert camera intrinsics to numpy array
    intrinsic_camera_matrix = np.array(camera_info_msg.K).reshape((3,3))
    
    # Convert image messages to cv2 images
    bridge = CvBridge() 
    rgb_image = bridge.imgmsg_to_cv2(image_msg[0], desired_encoding='passthrough')
    depth_image = bridge.imgmsg_to_cv2(image_msg[1], desired_encoding='passthrough')

    return rgb_image, depth_image, intrinsic_camera_matrix, frame_id




if __name__ == "__main__":
    rospy.init_node('RGBD_recorder', anonymous=False)
    record = False
    limit = 10 # seconds
    frame_num = 0
    done = False
    frame_rate = 30
    output_rgb_video = cv2.VideoWriter('rgb_easy.mp4',cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (1280,720))
    output_depth_video = cv2.VideoWriter('depth_easy.mp4',cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (1280,720))
    record = False
    limit = 10 # seconds
    done = False
    frame_rate = 20
    e = 0
    while not rospy.is_shutdown():
       
        # Request new message from rgbd_processor
        message = requestRGBD_client()
        rgb_image, depth_image, intrinsic_camera_matrix, frame_id = read_message(message)

        cv2.imshow("image", rgb_image)
        
        if cv2.waitKey(1) == ord('r'):
            record = True
            cv2.destroyAllWindows()
            s = time.time()
        while record:
            
            if e-s >= limit:
                record = False
                done = True

            # Request new message from rgbd_processor
            message = requestRGBD_client()
            rgb_image, depth_image, intrinsic_camera_matrix, frame_id = read_message(message)
            output_rgb_video.write(rgb_image)
            output_depth_video.write(depth_image)
            e = time.time()
        if done:
            output_depth_video.release()
            output_rgb_video.release()

            break
