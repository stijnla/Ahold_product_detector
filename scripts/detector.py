#!/usr/bin/env python3
import rospy
import cv2
from cv_bridge import CvBridge
import ultralytics
from ultralytics.yolo.utils.plotting import Annotator
import os
import numpy as np
from multi_object_tracker import Tracker

# message and service imports
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArrayWithCameraInfo
from ahold_product_detection.srv import *

class CameraData:
    def __init__(self) -> None:
        # Setup ros subscribers and service
        self.depth_subscriber = rospy.Subscriber(
            "/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback
        )
        self.rgb_subscriber = rospy.Subscriber(
            "/camera/color/image_raw", Image, self.rgb_callback
        )
        self.pointcould_subscriber = rospy.Subscriber(
            "/camera/depth/color/points", PointCloud2, self.pointcloud_callback
        )
        self.intrinsics_subscriber = rospy.Subscriber(
            "/camera/color/camera_info", CameraInfo, self.intrinsics_callback
        )

        self.bridge = CvBridge()

    def extrinsics_callback(self, data):
        # Get camera extrinsics from topic
        self.extrinsics = data

    def intrinsics_callback(self, data):
        # Get camera intrinsics from topic
        self.intrinsics = np.array(data.K).reshape((3, 3))

    def depth_callback(self, data):
        self.depth_msg = data

    def pointcloud_callback(self, data):
        self.pointcloud_msg = data

    def rgb_callback(self, data):
        self.rgb_msg = data

    @property
    def data(self):
        time_stamp = self.rgb_msg.header.stamp
        rgb_image = self.bridge.imgmsg_to_cv2(self.rgb_msg, desired_encoding="bgr8")
        depth_image = self.bridge.imgmsg_to_cv2(self.depth_msg, desired_encoding="passthrough")
        pointcloud = self.pointcloud_msg
        intrinsics = self.intrinsics

        # TODO: timesync or check if the time_stamps are not too far apart (acceptable error)

        return rgb_image, depth_image, pointcloud, time_stamp, intrinsics


class ProductDetector:
    def __init__(self) -> None:
        self.camera = CameraData()
        self.rate = rospy.Rate(30)
        weight_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "yolo_model", "best.pt"
        )
        self.model = ultralytics.YOLO(weight_path)

        # Initialize kalman filter for object tracking
        self.tracker = Tracker(
            dist_threshold=2,
            max_frame_skipped=5,
            max_trace_length=3,
            frequency=30,
            robot=False,
        )

    def plot_detection_results(self, frame, results):
        for r in results:
            annotator = Annotator(frame)

            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[
                    0
                ]  # get box coordinates in (top, left, bottom, right) format
                c = box.cls
                annotator.box_label(b, self.model.names[int(c)])

        frame = annotator.result()

        cv2.imshow("Result", frame)
        cv2.waitKey(1)

    def generate_detection_message(self, results, camera_intrinsics, time_stamp):
        # Get resulting bounding boxes defined as (x, y, width, height)
        rgb_bounding_boxes = results[0].boxes.xywh.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        bbox_msgs = []
        for i, bbox in enumerate(rgb_bounding_boxes):
            bbox_msg = BoundingBox()
            bbox_msg.pose.position.x, bbox_msg.pose.position.y, bbox_msg.pose.position.z = (bbox[0], bbox[1], 0)
            bbox_msg.pose.orientation.x, bbox_msg.pose.orientation.y, bbox_msg.pose.orientation.z, bbox_msg.pose.orientation.w = (0, 0, 0, 1)
            bbox_msg.dimensions.x, bbox_msg.dimensions.y, bbox_msg.dimensions.z = (bbox[2], bbox[3], 0)
            bbox_msg.value = scores[i]
            bbox_msg.label = classes[i]
            bbox_msgs.append(bbox_msg)

        detection_results_msg = BoundingBoxArrayWithCameraInfo()
        detection_results_msg.header.stamp = time_stamp
        detection_results_msg.boxes = bbox_msgs
        detection_results_msg.camera_info = camera_intrinsics
        return detection_results_msg
    

    def run(self):
        try:
            rgb_image, depth_image, pointcloud, time_stamp, camera_intrinsics = self.camera.data
        except Exception as e:
            print(e)
            return

        # predict
        results = self.model.predict(
            source=rgb_image,
            show=False,
            save=False,
            verbose=False,
            device=0,
        )

        detection_results_msg = self.generate_detection_message(results, camera_intrinsics, time_stamp)

        rgb_bounding_boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        # Relate rgb_bounding_boxes to depth image
        width_scale = depth_image.shape[1] / rgb_image.shape[1]
        height_scale = depth_image.shape[0] / rgb_image.shape[0]
        scale = np.array([width_scale, height_scale, width_scale, height_scale])
        depth_bounding_boxes = rgb_bounding_boxes.copy()
        depth_bounding_boxes[:, :4] *= scale
        depth_bounding_boxes = depth_bounding_boxes.astype(int)

        # Convert to 3D
        xyz_detections = []
        for depth_bounding_box in depth_bounding_boxes:
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
            scaled_xyz_vector = np.linalg.inv(self.camera.intrinsics) @ pixel_vector.T
            orientation = [0, 0, 0]
            xyz_detections.append(list(median_z * scaled_xyz_vector) + orientation)

        

        # Track the detected products with Kalman Filter
        self.tracker.process_detections(xyz_detections, classes, scores)

        self.plot_detection_results(rgb_image, results)

import time

if __name__ == "__main__":
    rospy.init_node("product_detector")
    detector = ProductDetector()
    t0 = time.time()
    while True:
        detector.run()
        detector.rate.sleep()
        print(f"fps {1/(time.time() - t0)}")
        t0 = time.time()
