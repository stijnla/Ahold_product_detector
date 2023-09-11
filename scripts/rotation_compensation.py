#!/usr/bin/env python3
import rospy
import tf
import numpy as np
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import cv2


class RotationCompensation():
    def __init__(self) -> None:
        self.listener = tf.TransformListener()
    
    def rotate_images(self, rgb_image, depth_image, stamp):
        # look up transformation and rotation matrix at time of image received TODO: time stamp requires extrapolation to the past???
        self.listener.waitForTransform('base_link', 'camera_color_optical_frame', rospy.Time(0), rospy.Duration(1.0))
        (trans,rot) = self.listener.lookupTransform('base_link', 'camera_color_optical_frame', rospy.Time(0))
        _, self.phi, _ = euler_from_quaternion(rot)
        rotated_rgb_image = self._rotate(rgb_image, -180*self.phi/(np.pi))
        rotated_depth_image = self._rotate(depth_image, -180*self.phi/(np.pi))
        return rotated_rgb_image, rotated_depth_image

    
    def _rotate(self, image, angle): 
        """Rotates image, and fills background with black or returns the largest rectangle possible inside rotated image"""
        self.rotation_matrix = cv2.getRotationMatrix2D((image.shape[1]/2,image.shape[0]/2),angle,1) 

        rotated_image = cv2.warpAffine(image,self.rotation_matrix,(image.shape[1],image.shape[0])) 
       
        return rotated_image

    def rotate_bounding_boxes(self, boxes_xywh, image, phi=0):
        centers = boxes_xywh[:, 0:2]
        dimensions = boxes_xywh[:, 2:4]
        
        if hasattr(self, 'phi'):
            angle = 180*self.phi/np.pi
        else:
            angle = 180*phi/np.pi

        
        rot_mat = cv2.getRotationMatrix2D((image.shape[1]/2,image.shape[0]/2), angle, 1) 
        
        centers_rotated = (rot_mat @ np.concatenate((centers, np.ones((centers.shape[0],1))), axis=1).T).T[:,:2]    
        if hasattr(self, 'phi'):  
            return np.concatenate((centers_rotated, dimensions), axis=1), self.phi
        else:
            return np.concatenate((centers_rotated, dimensions), axis=1), phi
