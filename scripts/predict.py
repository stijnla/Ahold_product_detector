#!/usr/bin/env python3
import rospy
from ahold_product_detection.srv import *
from cv_bridge import CvBridge
import ultralytics
import os
import numpy as np
import pyrealsense2 as rs
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import Header, Int32
from ahold_product_detection.msg import FloatList, PointCloudList, ProductClass
from tf.transformations import quaternion_from_euler
import tf2_ros
import tf
from geometry_msgs.msg import TransformStamped, PoseStamped
from itertools import combinations



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



def predict(model, rgb_image):
    """Returns predicted bounding boxes, possible classes, and predicted scores for each bounding box"""
    
    # Perform object detection on an image using the model (YOLO Neural Network)
    results = model.predict(source=rgb_image, show=True, save=False, verbose=False)

    # Get resulting bounding boxes defined as (x, y, width, height)
    bounding_boxes = results[0].boxes.xyxy.cpu().numpy()
    
    # Get confidence values (scores) of each bounding box
    scores = results[0].boxes.conf.cpu().numpy()

    # Get names of all classes that can be detected
    names = results[0].boxes.cls.cpu().numpy()

    return bounding_boxes, names, scores



def translate_rgb_bounding_boxes_to_depth(rgb_image, depth_image, xyxy_bounding_boxes):
    """returns bounding boxes translated to the depth image"""

    # Depth is aligned with rgb, but sizes differ    
    # cv2 image shapes are defined as (height, width, channels)
    width_scale = depth_image.shape[1] / rgb_image.shape[1] 
    height_scale = depth_image.shape[0] / rgb_image.shape[0]
    
    depth_bounding_boxes = []
    
    for bounding_box in xyxy_bounding_boxes:

        # Resize bounding box to depth image
        x_start = int(width_scale * bounding_box[0])
        x_end = int(width_scale * bounding_box[2])
        y_start = int(height_scale * bounding_box[1])
        y_end = int(height_scale * bounding_box[3])

        depth_bounding_box = np.array([x_start, y_start, x_end, y_end])

        depth_bounding_boxes.append(depth_bounding_box)
    
    return depth_bounding_boxes



def get_median_distance_depth_data(depth_data):
    """Returns median of depth data in meters"""
    mean_z = np.median(depth_data)/1000 #
    return mean_z



def get_grasp_coordinates(depth_image, depth_bounding_box, intrinsic_camera_matrix):
    """Returns xyz vector of the center of the bounding box (which is at the surface of the object)"""

    # Get depth data bounding box
    depth_data_bounding_box = depth_image[depth_bounding_box[1]:depth_bounding_box[3], depth_bounding_box[0]:depth_bounding_box[2]]

    # Get median of bounding box as estimated z coordinate
    median_z = get_median_distance_depth_data(depth_data_bounding_box)

    # Get bounding box center pixels
    bbox_center_u = int((depth_bounding_box[2] + depth_bounding_box[0])/2)
    bbox_center_v = int((depth_bounding_box[3] + depth_bounding_box[1])/2)

    # Calculate xyz vector with pixels (u, v) and camera intrinsics
    xyz_vector = translate_pixel_coordinates_to_cartesian_coordinates_in_camera_frame(bbox_center_u, bbox_center_v, median_z, intrinsic_camera_matrix)
    return xyz_vector



def translate_pixel_coordinates_to_cartesian_coordinates_in_camera_frame(u, v, estimated_z, intrinsic_camera_matrix):
    """Returns estimated xyz vector of each bounding box, representing the center of the bounding box at the object surface"""
    
    pixel_vector = np.array([u, v, 1])
    scaled_xyz_vector = np.linalg.inv(intrinsic_camera_matrix) @ pixel_vector.T 
    xyz_vector = estimated_z * scaled_xyz_vector

    return xyz_vector



def depth_bounding_box_to_pointcloud(depth_image, depth_bounding_box, intrinsic_camera_matrix, frame_id):
    """Returns pointcloud and pointcloud message, created from the depth data of the bounding box"""

    # Get depth data of bounding box
    product_depth_data =  depth_image[depth_bounding_box[1]:depth_bounding_box[3], depth_bounding_box[0]:depth_bounding_box[2]]

    # Create grid of all pixels for efficient matrix calculation
    uu, vv = np.meshgrid(np.arange(depth_bounding_box[0], depth_bounding_box[2]), np.arange(depth_bounding_box[1], depth_bounding_box[3]))
    uv_vector = np.vstack((uu.flatten(), vv.flatten(), np.ones(len(vv.flatten()))))

    # Get all z values that correspond with the pixels, format them to meters
    z_values = product_depth_data.flatten()/1000

    # Calculate pointcloud
    scaled_points = np.linalg.inv(intrinsic_camera_matrix) @ uv_vector
    points = z_values * scaled_points
    
    # Only consider points that have depth data != 0 (some depth data is inaccurate)
    nonzero_points = points.T[np.where(z_values!=0)]
    nonzero_z_values = z_values.T[np.where(z_values!=0)]

    # Only consider points that are close to the median of the pointcloud,
    # because the bounding box also includes some background points, which 
    # must be removed to include object points only
    bound = 0.025     # meters

    lower_bound = np.where(nonzero_z_values > np.median(nonzero_z_values) - bound, 1, 0)
    upper_bound = np.where(nonzero_z_values < np.median(nonzero_z_values) + bound, 1, 0)

    band_pass = lower_bound * upper_bound
    band_pass_z_values = band_pass * nonzero_z_values

    filtered_z_values = band_pass_z_values.T[np.where(band_pass_z_values!=0)]
    filtered_points = nonzero_points[np.where(band_pass_z_values!=0)]

    pointcloud = filtered_points
    
    # Generate pointcloud ROS message for visualization
    pointcloud_message = convert_PointCloud2_from_numpy_array(np.hstack((filtered_points, filtered_z_values.reshape(len(filtered_z_values), 1))), frame_id)

    return pointcloud, pointcloud_message



def estimate_pointcloud_orientation_with_plane(pointcloud):
    """Returns the orientation of pointcloud by fitting a plane to it"""
    fit = fit_plane(pointcloud)
    orientation = get_orientation_plane(fit)
    return orientation



def fit_plane(points):
    """Returns function values of plane fitted to a set of points (pointcloud)"""
    # Fit plane to function: ax + by + c = z (so goal: get a, b and c)
    A = np.hstack((points[:,:2], np.ones((len(points), 1)))) # xy1 vectors (= [x, y, 1])
    b = points[:, 2] # z values

    fit = np.linalg.pinv(A) @ b # [a, b, c]

    # Calculate error
    errors = b - A @ fit
    residual = np.linalg.norm(errors)

    return fit



def get_orientation_plane(fit):
    """Returns euler angles of a fitted plane"""
    theta = np.arctan(fit[1]) # rotation around camera x axis
    phi = np.arctan(fit[0])  # rotation around camera y axis
    
    return [float(theta), float(-phi), float(0)] # [x, y, z] rotations



def convert_PointCloud2_from_numpy_array(array, frame_id):
    """Returns PointCloud2 object made from a numpy array"""
    array = array.astype(np.float32)
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
              PointField('y', 4, PointField.FLOAT32, 1),
              PointField('z', 8, PointField.FLOAT32, 1),
              PointField('intensity', 12, PointField.FLOAT32, 1)]

    header = Header()
    header.frame_id = frame_id
    header.stamp = rospy.Time.now()

    pc2 = point_cloud2.create_cloud(header, fields, array)
    return pc2



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



def estimate_pose_object(depth_image, depth_bounding_box, intrinsic_camera_matrix, frame_id):
    """Returns pose message and pointcloud message"""
    # Calculate middle of bounding box (which is the center of the object's surface)
    xyz_vector = get_grasp_coordinates(depth_image, depth_bounding_box, intrinsic_camera_matrix)
    
    # Estimate the pointcloud that corresponds to the object's surface only
    pointcloud, pointcloud_message = depth_bounding_box_to_pointcloud(depth_image, depth_bounding_box, intrinsic_camera_matrix, frame_id)
    
    # Estimate orientation of object using a plane fit around the object's surface pointcloud
    orientation = estimate_pointcloud_orientation_with_plane(pointcloud)
    
    pose_message = FloatList()
    vector = xyz_vector.tolist() + orientation
    pose_message.data = vector

    return pose_message, pointcloud_message



def estimate_pose_detected_objects(rgb_image, depth_image, rgb_bounding_boxes, intrinsic_camera_matrix, frame_id):
    """Returns the poses and pointclouds of all detected objects"""
    pointcloud_messages = []
    object_poses = []
    
    # Relate rgb_bounding_boxes to depth image
    depth_bounding_boxes = translate_rgb_bounding_boxes_to_depth(rgb_image, depth_image, rgb_bounding_boxes)
    
    for depth_bounding_box in depth_bounding_boxes:
        pose_message, pointcloud_message = estimate_pose_object(depth_image, 
                                                                depth_bounding_box, 
                                                                intrinsic_camera_matrix, 
                                                                frame_id)
        
        object_poses.append(pose_message)
        pointcloud_messages.append(pointcloud_message)
    
    return object_poses, pointcloud_messages



def find_desired_object(desired_object_index, names, object_poses, scores, pointcloud_messages):
    """Returns all objects that are desired, along with their scores and pointclouds"""
    desired_objects = []
    desired_object_scores = []
    desired_pointclouds = []


    for i, name in enumerate(names):
        if name == desired_object_index:
            
            desired_objects.append(object_poses[i])
            desired_object_scores.append(scores[i])
            desired_pointclouds.append(pointcloud_messages[i])
    
    return desired_objects, desired_object_scores, desired_pointclouds



def broadcast_tf_transform_of_object(object_pose, index, timeStamp, listener):
    """Converts message to a tf2 frame when message becomes available, and then broadcasts it"""
    br = tf2_ros.TransformBroadcaster()
    

    x, y, z, theta, phi, psi = object_pose.data
    
    robot = False
    if robot:
        p = PoseStamped()
        p.header.stamp = timeStamp
        rospy.logwarn(timeStamp)
        p.header.frame_id = "camera_color_optical_frame"
        #t.child_frame_id = 'object'+str(index)
        p.pose.position.x = x
        p.pose.position.y = y
        p.pose.position.z = z
        q = quaternion_from_euler(theta, phi, 0)
        p.pose.orientation.x = q[0]
        p.pose.orientation.y = q[1]
        p.pose.orientation.z = q[2]
        p.pose.orientation.w = q[3]

        p_transformed = listener.transformPose('base_link', p)

        t = TransformStamped()

        t.header.stamp = timeStamp
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'object'+str(index)
        t.transform.translation.x = p_transformed.pose.position.x
        t.transform.translation.y = p_transformed.pose.position.y
        t.transform.translation.z = p_transformed.pose.position.z
        t.transform.rotation.x = p_transformed.pose.orientation.x
        t.transform.rotation.y = p_transformed.pose.orientation.y
        t.transform.rotation.z = p_transformed.pose.orientation.z
        t.transform.rotation.w = p_transformed.pose.orientation.w
        
    else:
        t = TransformStamped()
        t.header.stamp = timeStamp
        t.header.frame_id = 'camera_color_optical_frame'
        t.child_frame_id = 'object'+str(index)
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z
        q = quaternion_from_euler(theta, phi, 0)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
    br.sendTransform(t)



def calculate_distance_poses(pose1, pose2):
    return np.sqrt((pose1[0] - pose2[0])**2 + (pose1[1] - pose2[1])**2 + (pose1[2] - pose2[2])**2)



def search_products(message, model):
    # Convert message to usable data
    rgb_image, depth_image, intrinsic_camera_matrix, frame_id = read_message(message)

    # Predict product bounding boxes and classes
    rgb_bounding_boxes, names, scores = predict(model, rgb_image)
    
    object_poses, pointcloud_messages = estimate_pose_detected_objects(rgb_image, 
                                                                    depth_image, 
                                                                    rgb_bounding_boxes, 
                                                                    intrinsic_camera_matrix, 
                                                                    frame_id)
    return object_poses, pointcloud_messages, names, scores



def broadcast_detected_products(object_poses, names, scores, pub_object_class, listener, num_objects):
    if len(object_poses) > num_objects:
        new_num_objects = len(object_poses)
    else:
        new_num_objects = num_objects

    for i in range(new_num_objects):

        try:
            object_pose = object_poses[i]
            timeStamp = rospy.Time.now()
            
            object_class_message = ProductClass()
            object_class_message.classification = int(names[i])
            object_class_message.score = scores[i]
            object_class_message.header.stamp = timeStamp
            pub_object_class.publish(object_class_message)
            broadcast_tf_transform_of_object(object_pose, i, timeStamp, listener)

        except:
            empty_pose = FloatList()
            empty_pose.data = [0, 0, 0, 0, 0, 0]
            broadcast_tf_transform_of_object(empty_pose, i, rospy.Time.now(), listener)

    return new_num_objects
    


def main():
    # Load yolo model weights for detection
    weight_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'yolo_model', 'nano_supermarket_best.pt')
    model = ultralytics.YOLO(weight_path)

    # Initialize publishers for intermediate results visualization
    pub_num_objects = rospy.Publisher("number_of_objects", Int32, queue_size=1)
    pub_object_class = rospy.Publisher("detected_classes", ProductClass, queue_size=1)

    rospy.init_node('Product_detector', anonymous=False)

    listener = tf.TransformListener()
    rospy.sleep(1)
    num_objects = 0


    while not rospy.is_shutdown():
        # Receive message from rgbd_processor
        message = requestRGBD_client()
        
        if message != None:
            object_poses, pointcloud_messages, names, scores = search_products(message, model)
            
            num_objects = broadcast_detected_products(object_poses, names, scores, pub_object_class, listener, num_objects)
                
            pub_num_objects.publish(num_objects)
            
            
            


if __name__ == '__main__':
    main()