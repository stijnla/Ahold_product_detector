#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
import realsense2_camera as rs
from cv_bridge import CvBridge
import cv2
from ahold_product_detection.srv import RequestRGBD, RequestRGBDResponse


class ProcessCameraData:

    def __init__(self) -> None:

        # Setup ros subscribers and service
        self.depth_subscriber = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        self.rgb_subscriber = rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_callback)
        self.pointcould_subscriber = rospy.Subscriber("/camera/depth/color/points", PointCloud2, self.pointcloud_callback)
        self.intrinsics_subscriber = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.intrinsics_callback)
        self.rgbd_service = rospy.Service("request_rgbd", RequestRGBD, self.handle_image_request)
        
        # Keep the last 10 data points in a cache
        cache_size = 10
        self.depth_images_cache = [None]*cache_size
        self.rgb_images_cache = [None]*cache_size
        self.pointcloud_cache = [None]*cache_size
        self.intrinsics = None
        self.extrinsics = None
        self.setup_complete = False
        # Turn image message in cv2 image object
        self.bridge = CvBridge()
    


    def handle_image_request(self, req):
        # Send most recent data when requested

        try:
            rgb_image = self.rgb_images_cache[0]
            rgb_msg = self.bridge.cv2_to_imgmsg(rgb_image)

            depth_image = self.depth_images_cache[0]
            depth_msg = self.bridge.cv2_to_imgmsg(depth_image)

            pointcloud = self.pointcloud_cache[0]
            
            rospy.loginfo("Satisfied data request")
            
            return RequestRGBDResponse(images=[rgb_msg, depth_msg], pointcloud=pointcloud, intrinsics=self.intrinsics, data_available=self.setup_complete)#, extrinsics=self.extrinsics)
        
        except:
            rospy.logwarn("No camera data received yet!")

    

    def update_cache(self, cache, data):
        # Move all stored data points up one position
        cache = [cache[i] for i, _ in enumerate(cache) if i < len(cache) - 1]
        
        # Insert new data in first position of cache
        cache.insert(0, data)

        return cache
    


    def extrinsics_callback(self, data):
        # Get camera extrinsics from topic
        self.extrinsics = data


    
    def intrinsics_callback(self, data):
        # Get camera intrinsics from topic
        self.intrinsics = data



    def depth_callback(self, data):
        # Get and convert depth data from topic
        depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        self.depth_images_cache = self.update_cache(self.depth_images_cache, depth_image)
        self.setup_complete = True

    
    def pointcloud_callback(self, data):
        # Get pointcloud data from topic
        pointcloud = data
        self.pointcloud_cache = self.update_cache(self.pointcloud_cache, pointcloud)
        self.setup_complete = True
    

    def rgb_callback(self, data):
        # Get and convert rgb data from topic
        bgr_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        self.rgb_images_cache = self.update_cache(self.rgb_images_cache, rgb_image)
        self.setup_complete = True


def main():
    rospy.init_node('RGBD-Processor', anonymous=False)
    ProcessCameraData()
    rospy.spin()
            
if __name__ == '__main__':
    main()