#!/usr/bin/env python3
import rospy
from ahold_product_detection.msg import Detection, ProductPose, ProductPoseArray
import tf
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Pose, Vector3
import numpy as np
from cv_bridge import CvBridge
from copy import copy, deepcopy
import tf
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from sensor_msgs.msg import CameraInfo
from rotation_compensation import rotate_image, rotate_bounding_boxes
from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox
from ahold_product_detection.srv import ChangeProduct, ChangeProductResponse
import tf2_ros
from geometry_msgs.msg import TransformStamped

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


class DetectorData:
    def __init__(self) -> None:
        self.subscriber = rospy.Subscriber(
            "/detection_results", Detection, self.callback
        )
        self.data = None
        self.bridge = CvBridge()
        self.previous_stamp = rospy.Time.now()

    def callback(self, data):
        self.data = data


class PoseEstimator:
    def __init__(self) -> None:
        self.detection = DetectorData()
        self.rate = rospy.Rate(30)
        self.tf_listener = tf.TransformListener()
        self.requested_yolo_id = -1
        self.closed_loop = True
        self.first_transform_sent = False
        self.requested_product_tracked = False
        self.pub_results = rospy.Publisher(
            "/pose_estimation_results", ProductPoseArray, queue_size=10
        )
        self.pub_detections = rospy.Publisher(
            "/3d_detections", BoundingBoxArray, queue_size=10
        )

        self.intrinsics_subscriber = rospy.Subscriber(
            "/camera/color/camera_info", CameraInfo, self.intrinsics_callback
        )
        rospy.sleep(1.0)

    def intrinsics_callback(self, data):
        self.intrinsics = np.array(data.K).reshape((3, 3))

    def transform_poses(self, product_poses):
        transformed_poses = ProductPoseArray()
        transformed_poses.header = product_poses.header
        for pose in product_poses.poses:
            ros_pose = PoseStamped()
            ros_pose.header.stamp = product_poses.header.stamp
            ros_pose.header.frame_id = "camera_color_optical_frame"
            ros_pose.pose.position = Point(x=pose.x, y=pose.y, z=pose.z)
            ros_pose.pose.orientation = Quaternion(
                *quaternion_from_euler(pose.theta, pose.phi, pose.psi)
            )
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
            new_pose.theta, pose.phi, pose.psi = euler_from_quaternion(
                list(ros_pose_transformed.pose.orientation)
            )
            new_pose.header = pose.header
            new_pose.header.frame_id = "base_link"
            new_pose.label = pose.label
            new_pose.score = pose.score
            transformed_poses.poses.append(new_pose)

        return transformed_poses

    def estimate_bbox_depth(self, boxes, depth_image):
        rotated_depth_image = rotate_image(image=depth_image, angle=boxes[0].angle)
        boxes_xywh = np.array([[box.x, box.y, box.w, box.h] for box in boxes])
        rotated_bounding_boxes = rotate_bounding_boxes(
            boxes_xywh, depth_image, angle=boxes[0].angle
        )

        estimated_z = []
        for x, y, w, h in rotated_bounding_boxes:
            depth_path = rotated_depth_image[
                int(y - (w / 2)) : int(y + (w / 2)), int(x - (h / 2)) : int(x + (h / 2))
            ]
            estimated_z.append(np.median(depth_path[depth_path != 0])/1000)

        return estimated_z

    def estimate_pose_bounding_box(self, z, bbox):
        pixel_vector = np.array([bbox.x, bbox.y, 1])
        scaled_xyz_vector = np.linalg.inv(self.intrinsics) @ pixel_vector.T
        orientation = [0, 0, 0]

        return list(z * scaled_xyz_vector) + orientation

    def change_product_cb(self, request):
        rospy.loginfo(f"Changing tracked product from {self.tracker.requested_yolo_id} to {request.product_id}")
        self.requested_yolo_id = request.product_id
        return ChangeProductResponse(success=True)
    
    def broadcast_product_to_grasp(self, product_to_grasp):
        # Convert message to a tf2 frame when message becomes available
        br = tf2_ros.TransformBroadcaster()
        t = TransformStamped()

        x, y, z, theta, phi, psi = np.array(product_to_grasp.KF.state[:6])

        t.header.stamp = rospy.Time.now()

        if self.robot:
            t.header.frame_id = "base_link"
        else:
            t.header.frame_id = "camera_color_optical_frame"
        t.child_frame_id = 'desired_product'
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z
        q = quaternion_from_euler(theta, phi,  np.pi/2)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        br.sendTransform(t)

    def broadcast_static_product_to_grasp(self, product_to_grasp):
        # Convert message to a tf2 frame when message becomes available
        br = tf2_ros.StaticTransformBroadcaster()
        t = TransformStamped()

        x, y, z, theta, phi, psi = np.array(product_to_grasp.KF.state[:6])

        t.header.stamp = rospy.Time.now()

        if self.robot:
            t.header.frame_id = "base_link"
        else:
            t.header.frame_id = "camera_color_optical_frame"
        t.child_frame_id = 'desired_product'
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z
        q = quaternion_from_euler(theta, phi,  np.pi/2)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        br.sendTransform(t)




    def choose_desired_product_score(self, poses):
        desired_product = self.requested_yolo_id 

        # product not yet chosen OR not tracked anymore, choose product that is most often classified as desired product
        scores = np.array([pose.score for pose in poses if pose.label == desired_product])
        scores_detection_indices = np.array([i for i, pose in enumerate(poses) if pose.label == desired_product])
        if len(scores) == 0:
            return None
        
        self.index_product_to_grasp = self.tracks[scores_detection_indices[np.argmax(scores)]].track_id
        return self.tracks[scores_detection_indices[np.argmax(scores)]]
    

    def run(self):
        # read data when available
        try:
            
            depth_image = self.detection.bridge.imgmsg_to_cv2(
                self.detection.data.depth_image, desired_encoding="passthrough"
            )
            stamp = copy(self.detection.data.header.stamp)
            boxes = self.detection.data.detections

            
            if self.detection.previous_stamp.to_sec() == stamp.to_sec():
                raise ValueError("New data has not been received... stopping pose estimation")
            self.detection.previous_stamp = stamp

        except Exception as e:
            return

        if len(boxes) == 0:
            return

        # Convert to 3D
        product_poses = ProductPoseArray()
        product_poses.header.stamp = stamp

        boxes_estimated_z = self.estimate_bbox_depth(boxes, depth_image)
        for z, bbox in zip(boxes_estimated_z, boxes):
            if not np.isnan(z):
                # depth exists and non-zero
                xyz_detection = self.estimate_pose_bounding_box(z, bbox)
                product_pose = ProductPose()

                (
                    product_pose.x,
                    product_pose.y,
                    product_pose.z,
                    product_pose.theta,
                    product_pose.phi,
                    product_pose.psi,
                ) = xyz_detection
                product_pose.score = bbox.score
                product_pose.label = bbox.label

                product_poses.poses.append(product_pose)

        # Transform to non-moving frame
        transformed_product_poses = self.transform_poses(product_poses)

        # Create jsk boundingboxes
        detections = BoundingBoxArray()
        detections.header.frame_id = "base_link"
        for product_pose in transformed_product_poses.poses:
            detection = BoundingBox()
            detection.header.frame_id = "base_link"
            detection.pose = Pose(
                position=Point(x=product_pose.x, y=product_pose.y, z=product_pose.z),
                orientation=Quaternion(
                    *quaternion_from_euler(
                        product_pose.theta, product_pose.phi, product_pose.psi
                    )
                )
            )
            detection.dimensions = Vector3(x=0.05, y=0.05, z=0.05)
            detection.value = product_pose.score
            detection.label = product_pose.label

            detections.boxes.append(detection)


        #########################################################################
        #########################################################################
        #########################################################################
        # Change to desired choosing method (occurance or score based)
        product_to_grasp = self.choose_desired_product_score(transformed_product_poses.poses)

        self.requested_product_tracked = product_to_grasp != None
        if self.requested_product_tracked:
            if self.closed_loop:
                self.broadcast_product_to_grasp(product_to_grasp)
            else:
                if not self.first_transform_sent:
                    self.broadcast_static_product_to_grasp(product_to_grasp)
                    self.first_transform_sent = True
        #########################################################################
        #########################################################################
        #########################################################################

        self.pub_detections.publish(detections)


import time

if __name__ == "__main__":
    rospy.init_node("product_pose_estimator")
    pose_estimator = PoseEstimator()

    t0 = time.time()
    while not rospy.is_shutdown():
        pose_estimator.run()
        pose_estimator.rate.sleep()
        # print(f"product pose estimation rate: {1/(time.time() - t0)}")
        t0 = time.time()
