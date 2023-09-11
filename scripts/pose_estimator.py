#!/usr/bin/env python3
import rospy
from ahold_product_detection.msg import Detection, ProductPose, ProductPoseArray
import tf
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import PoseStamped, Point, Quaternion
import numpy as np
from cv_bridge import CvBridge
from copy import copy


# Helpers
def _it(self):
    yield self.x
    yield self.y
    yield self.z
Point.__iter__ = _it


def _it(self):
    yield self.x
    yield self.y
    yield self.z
    yield self.w
Quaternion.__iter__ = _it


class DetectorData():
    def __init__(self) -> None:
        self.subscriber = rospy.Subscriber("/detection_results", Detection, self.callback)
        self.data = None
        self.bridge = CvBridge()

    def callback(self, data):
        self.data = data

class PoseEstimator():
    def __init__(self) -> None:
        self.detection = DetectorData()
        self.rate = rospy.Rate(30)
        self.tf_listener = tf.TransformListener()
        self.pub = rospy.Publisher('/pose_estimation_results', ProductPoseArray, queue_size=10)
    
    def transform_poses(self, product_poses):
        transformed_poses = ProductPoseArray()
        transformed_poses.header = product_poses.header
        for pose in product_poses.poses:
            ros_pose = PoseStamped()
            ros_pose.header.stamp = product_poses.header.stamp
            ros_pose.header.frame_id = "camera_color_optical_frame"
            ros_pose.pose.position = Point(x=pose.x, y=pose.y, z=pose.z)
            ros_pose.pose.orientation = Quaternion(*quaternion_from_euler(pose.theta, pose.phi, pose.psi))
            try:
                ros_pose_transformed = self.tf_listener.transformPose(
                    "base_link", ros_pose
                )
            except Exception as e:
                rospy.logerr("couldn't transform correctly ", e)

            new_pose = ProductPose()
            new_pose.x = ros_pose_transformed.pose.position.x
            new_pose.y = ros_pose_transformed.pose.position.y
            new_pose.z = ros_pose_transformed.pose.position.z
            new_pose.theta, pose.phi, pose.psi = euler_from_quaternion(list(ros_pose_transformed.pose.orientation))
            new_pose.header = pose.header
            new_pose.header.frame_id = "base_link"
            new_pose.label = pose.label
            new_pose.score = pose.score
            transformed_poses.poses.append(new_pose)



        return transformed_poses

    def run(self):
        # read data when available
        try:
            depth_image = self.detection.bridge.imgmsg_to_cv2(self.detection.data.depth_image, desired_encoding="passthrough")
            rgb_image = self.detection.bridge.imgmsg_to_cv2(self.detection.data.rgb_image, desired_encoding="bgr8")
            bounding_boxes = self.detection.data.boundingboxarraywithcamerainfo.boxes
            camera_intrinsics = np.array(self.detection.data.boundingboxarraywithcamerainfo.camera_info.K).reshape((3, 3))
        except Exception as e:
            return
        
        rgb_bounding_boxes = []
        scores = []
        labels = []

        # read bounding box data
        for box in bounding_boxes.boxes:
            #new_box  = np.array([box.pose[0], box.pose[1], box.dimensions[0], box.dimensions[1]]) #xywh
            new_box  = np.array([box.pose.position.x - box.dimensions.x/2, box.pose.position.y - box.dimensions.y/2, box.pose.position.x + box.dimensions.x/2, box.pose.position.y + box.dimensions.y/2]) #xywh
            score = box.value
            label = box.label
            rgb_bounding_boxes.append(new_box)
            scores.append(score)
            labels.append(label)

        
        if rgb_bounding_boxes != []:
            
            # Relate rgb_bounding_boxes to depth image
            """
            rospy.logwarn(str(depth_image.shape) + ' = ' + str(rgb_image.shape))
            width_scale = depth_image.shape[1] / rgb_image.shape[1]
            height_scale = depth_image.shape[0] / rgb_image.shape[0]
            scale = np.array([width_scale, height_scale, width_scale, height_scale])
            depth_bounding_boxes = np.array(copy(rgb_bounding_boxes))
            depth_bounding_boxes[:, :4] *= scale
            depth_bounding_boxes = depth_bounding_boxes.astype(int)
            """

            depth_bounding_boxes = np.array(copy(rgb_bounding_boxes))
            
            # Convert to 3D
            xyz_detections = []
            product_poses = ProductPoseArray()
            product_poses.header.stamp = self.detection.data.header.stamp

            for i, depth_bounding_box in enumerate(depth_bounding_boxes):
                product_pose = ProductPose()

                # Get depth data bounding box
                depth_data_bounding_box = depth_image[
                    int(depth_bounding_box[1]) : int(depth_bounding_box[3]),
                    int(depth_bounding_box[0]) : int(depth_bounding_box[2]),
                ]

                median_z = np.median(depth_data_bounding_box) / 1000

                # Get bounding box center pixels
                bbox_center_u = int((depth_bounding_box[2] + depth_bounding_box[0]) / 2)
                bbox_center_v = int((depth_bounding_box[3] + depth_bounding_box[1]) / 2)

                # Calculate xyz vector with pixels (u, v) and camera intrinsics
                pixel_vector = np.array([bbox_center_u, bbox_center_v, 1])
                scaled_xyz_vector = np.linalg.inv(camera_intrinsics) @ pixel_vector.T
                orientation = [0, 0, 0]
                xyz_detection = list(median_z * scaled_xyz_vector) + orientation

                product_pose.x, product_pose.y, product_pose.z, product_pose.theta, product_pose.phi, product_pose.psi = xyz_detection
                product_pose.score = scores[i]
                product_pose.label = labels[i]

                product_poses.poses.append(product_pose)

                xyz_detections.append(xyz_detection)

            # Transform to non-moving frame
            transformed_poses = self.transform_poses(product_poses)

            self.pub.publish(transformed_poses)

import time

if __name__ == "__main__":
    rospy.init_node("product_pose_estimator")
    pose_estimator = PoseEstimator()

    
    t0 = time.time()
    while not rospy.is_shutdown():
        pose_estimator.run()
        pose_estimator.rate.sleep()
        print(f"product pose estimation rate: {1/(time.time() - t0)}")
        t0 = time.time()