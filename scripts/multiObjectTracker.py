#!/usr/bin/env python3
import rospy
import numpy as np
from kalmanFilter_position import KalmanFilter
from collections import deque
import time
import cv2
from scipy.optimize import linear_sum_assignment
import random
from ahold_product_detection.msg import FloatList, ProductList
from tf.transformations import quaternion_from_euler
import tf2_ros
from geometry_msgs.msg import TransformStamped



class rotatedRect():

    def __init__(self, frame, center, width, height, angle, color, thickness) -> None:
        self.center = center
        self.width = width
        self.height = height
        self.angle = angle
        self.determine_rect_corners()
        
        self.frame = frame

        self.color = color
        self.thickness = 2
        
        cv2.line(self.frame, self.corners[0], self.corners[1], self.color, self.thickness)
        cv2.line(self.frame, self.corners[1], self.corners[2], self.color, self.thickness)
        cv2.line(self.frame, self.corners[2], self.corners[3], self.color[::-1], self.thickness)
        cv2.line(self.frame, self.corners[3], self.corners[0], self.color, self.thickness)

    def determine_rect_corners(self):
        rot_mat = np.array([[np.cos(self.angle), -np.sin(self.angle)],
                            [np.sin(self.angle),  np.cos(self.angle)]])
        w = self.width/2
        h = self.height/2
        
        non_rotated_corner_vectors = np.array([[ w,  h],
                                   [-w,  h],
                                   [-w, -h],
                                   [ w, -h]])
        rotated_corner_vectors = rot_mat @ non_rotated_corner_vectors.T
        self.corners = np.array(rotated_corner_vectors.T, dtype=np.int32) + self.center




class Tracks:

    def __init__(self, measurement, classification, score, track_id, frequency):
        state = np.array(measurement) # initial state, speeds are both 0
        self.KF = KalmanFilter(init_state=state, frequency=frequency, measurement_variance=0.1)
        self.trace = deque(maxlen=20)
        self.track_id = track_id
        self.skipped_frames = 0
        self.classifications = []
        self.scores = []
        self.frequencies = []
        if classification == classification and score == score:
            self.classifications.append(classification)
            self.scores.append(score)


    @property
    def prediction(self):
        return np.array([self.KF.pred_state[0, 0], self.KF.pred_state[1, 0], self.KF.pred_state[2, 0], self.KF.pred_state[3, 0], self.KF.pred_state[4, 0], self.KF.pred_state[5, 0]])



    @property
    def classification(self):
        unique_classifications, counts = np.unique(self.classifications, return_counts=True)
        index = np.argmax(counts)
        
        return unique_classifications[index]
    


    @property
    def score(self):
        # EWMA WITH SCORES
        classification_indices = [i for i, _ in enumerate(self.classifications)]
        mean_score = sum([self.scores[i] for i in classification_indices]) / len(classification_indices)
        
        return mean_score

    
    @property
    def variance(self):
        return self.KF.pred_err_cov



    def update(self, measurement, classification, score, frequency):
        # Add classification to classifications
        if classification == classification and score == score:
            self.classifications.append(classification)
            self.scores.append(score)

        # Update the state transition matrix based on the passed time
        if self.frequencies == []:
            self.KF.update_matrices(frequency)
        else:
            freq = 1/(sum([1/f for f in self.frequencies]) + 1/frequency)
            self.KF.update_matrices(freq)

        # Update the state based on the new measurements
        self.KF.update(np.array(measurement).reshape(6, 1))



    def update_no_measurement(self, frequency):
        self.frequencies.append(frequency)
        freq = 1/sum([1/f for f in self.frequencies])
        self.KF.update_matrices(freq)


class Tracker:

    def __init__(self, dist_threshold, max_frame_skipped, max_trace_length, frequency):
        self.dist_threshold = dist_threshold
        self.max_frame_skipped = max_frame_skipped
        self.max_trace_length = max_trace_length
        self.current_track_id = 0
        self.tracks = []
        self.frequency = frequency
        self.skip_frame_count = 0
        self.previous_measurement_exists = False
        self.track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                            (127, 127, 255), (255, 0, 255), (255, 127, 255),
                            (127, 0, 255), (127, 0, 127),(127, 10, 255), (0,255, 127)]
        
        self.index_product_to_grasp = None
        self.initial_score_product_to_grasp = None
        



    def process_detections(self, data):
        detected_products = [msg.data for msg in data.data]
        product_poses = [detected_product[2::] for detected_product in detected_products]
        product_scores = [detected_product[1] for detected_product in detected_products]
        product_classes = [detected_product[0] for detected_product in detected_products]

        

        

        current_time = rospy.get_time()
        if self.previous_measurement_exists:
            delta_t = current_time - self.prev_time
        else:
            delta_t = 1
        
        self.previous_measurement_exists = True
        self.prev_time = current_time
        self.update(product_poses, product_classes, product_scores, 1/float(delta_t)) 
        product_to_grasp = self.choose_desired_product()
        if product_to_grasp != None:
            self.broadcast_product_to_grasp(product_to_grasp)
        self.visualize(product_poses, product_to_grasp)



    def broadcast_product_to_grasp(self, product_to_grasp):
        # Convert message to a tf2 frame when message becomes available
        br = tf2_ros.TransformBroadcaster()
        t = TransformStamped()

        x, y, z, theta, phi, psi = product_to_grasp.trace[-1]

        t.header.stamp = rospy.Time.now()
        robot = False
        if robot:
            t.header.frame_id = "base_link"
        else:
            t.header.frame_id = "camera_color_optical_frame"
        t.child_frame_id = 'desired_product'
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z
        q = quaternion_from_euler(theta, phi, 0)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        br.sendTransform(t)


        
    def visualize(self, measurements, product_to_grasp):
        frame_xz = self.draw_xz_view(measurements, product_to_grasp)        
        cv2.imshow('kalman filter xz', frame_xz)
        cv2.waitKey(1)



    def draw_xz_view(self, measurements, product_to_grasp):
        width = 600
        height = 600
        frame = createimage(height, width)

        cv2.circle(frame, (int(width/2), 0), 10, (0,0,255), 5)
        scale = 250 # convert millimeters to 0.25meters

        # Draw the measurements
        for measurement in measurements:

            x = -int(scale * measurement[0]) + int(width/2)
            z = int(scale *measurement[2])

            cv2.circle(frame, (x, z), 6, (0,0,0), -1)
        
        # Draw the latest updated states
        for track in self.tracks:
            updated_state = track.trace[-1]

            x = -int(scale * updated_state[0]) + int(width/2)
            z = int(scale *updated_state[2])
            theta = updated_state[4][0]

            variance_scale = 100000
            axis_length = (int(track.KF.pred_err_cov[0,0]/variance_scale), int(track.KF.pred_err_cov[2,2]/variance_scale))
            print(axis_length)
            rotatedRect(frame, (x, z), 20, 20, theta, (0,0,255), 3)
            cv2.ellipse(frame, (x, z), axis_length, 0, 0, 360, (0,255,255), 3)
            cv2.putText(frame, str(track.classification), (x + 5, z - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0,200,0), 1)
        
        # product to grasp
        if product_to_grasp != None:
            updated_state = product_to_grasp.trace[-1]
            x = -int(scale * updated_state[0]) + int(width/2)
            z = int(scale *updated_state[2])
            theta = updated_state[4][0]
            cv2.circle(frame, (x, z), 15, (0,0,0), 3)

        return frame



    def draw_xy_view(self, measurements):
        width = 600
        height = 600
        frame = createimage(height, width)

        scale = 1000 # convert millimeters to meters

        # Draw the measurements
        for measurement in measurements:

            x = -int(scale * measurement[0]) + int(width/2)
            y = int(scale *measurement[1]) + int(height/2)

            cv2.circle(frame, (x, y), 6, (0,0,0), -1)
        
        # Draw the latest updated states
        for track in self.tracks:
            updated_state = track.trace[-1]
            
            """
            x = -int(scale * updated_state[0]) + int(width/2)
            y = int(scale *updated_state[2]) + int(height/2)
            """

            x = -int(scale * updated_state[0]) + int(width/2)
            y = int(scale *updated_state[1]) + int(height/2)

            cv2.circle(frame, (x, y), 6, (0,255,0), 3)
        return frame
    


    def choose_desired_product(self):

        desired_product = 93
        minimun_required_detections = 10
        
        switch_threshold = 0.8
        detected_desired_product_scores = []
        for i, track in enumerate(self.tracks):

            if track.classification == desired_product and len(track.classifications) > minimun_required_detections:
                detected_desired_product_scores.append(track.score)
        
        if detected_desired_product_scores != []:
            if self.index_product_to_grasp == None:
                self.initial_score_product_to_grasp = max(detected_desired_product_scores)
                self.index_product_to_grasp = self.tracks[detected_desired_product_scores.index(self.initial_score_product_to_grasp)].track_id
            
        desired_track = []
        if self.index_product_to_grasp != None:
            for track in self.tracks:
                if track.track_id == self.index_product_to_grasp:
                    desired_track.append(track)
            
            if desired_track == []:
                self.index_product_to_grasp = None
                self.initial_score_product_to_grasp = None
            else:
                return desired_track[0]
        return None



    def calculate_variance_measurements(self, measurement):
        measurement = np.array(measurement).reshape(6,1)
        if not hasattr(self, 'mean'):
            self.mean = np.zeros((6, 1))
            self.num_measurements = 0
            self.measurement_variance = np.zeros((6, 6))

        # Update mean
        self.mean = (self.mean * self.num_measurements + measurement) / (self.num_measurements + 1)

        # Update number of measurements
        self.num_measurements = self.num_measurements + 1

        # Update measurement_variance
        np.set_printoptions(suppress = True)
        if self.num_measurements > 2:
            self.measurement_variance = ((measurement - self.mean) @ (measurement - self.mean).T + self.measurement_variance * (self.num_measurements - 2)) / (self.num_measurements - 1) 
        else:
            self.measurement_variance = ((measurement - self.mean) @ (measurement - self.mean).T) / (self.num_measurements - 1) 
        
        print(np.round((10**9)*self.measurement_variance, decimals=1))
        print(self.num_measurements)
    


    def update(self, measurements, classifications, scores, current_frequency):
        
        # Initialize tracks
        if len(self.tracks) == 0:
            for i, measurement in enumerate(measurements):
                self.tracks.append(Tracks(measurement, classifications[i], scores[i], self.current_track_id, current_frequency))
                self.current_track_id += 1
        
        if len(measurements) > 0:
            # Calculate distance measurements w.r.t. existing track predictions
            dists = np.array([np.linalg.norm(measurements - track.prediction, axis=1) for track in self.tracks])

            # Determine which measurement belongs to which track
            assignment = np.array(linear_sum_assignment(dists)).T
            
            # Only assign a measurement to a track if it is close enough to the predicted position
            assignment = [a for a in assignment if dists[a[0], a[1]] < self.dist_threshold]

            # Update state of existing tracks with measurement
            for track_idx, measurement_idx in assignment:
                self.tracks[track_idx].update(measurements[measurement_idx], classifications[measurement_idx], scores[measurement_idx], current_frequency)
                self.tracks[track_idx].skipped_frames = 0
                self.tracks[track_idx].frequencies = []
    
                #self.calculate_variance_measurements(measurements[measurement_idx])
            
            # Create new tracks for measurements without track
            assigned_det_idxs = [det_idx for _, det_idx in assignment]
            for i, det in enumerate(measurements):
                if i not in assigned_det_idxs:
                    # TODO: do not add tracks that are (0, 0, 0) (bad measurement)
                    self.tracks.append(Tracks(det, classifications[i], scores[i], self.current_track_id, current_frequency))
                    self.current_track_id += 1

            # Propagate unassigned tracks using the prediction 
            assigned_track_idxs = [track_idx for track_idx, _ in assignment]
            for i, track in enumerate(self.tracks):
                if i not in assigned_track_idxs:
                    # No measurement available, assume prediction is right
                    track.update_no_measurement(current_frequency)

                    # Keep track of the missed measurements of this object
                    track.skipped_frames += 1

        else:
            # No measurements, update tracks according to prediction
            for i, track in enumerate(self.tracks):
                # No measurement available, assume prediction is right
                #TODO: do not update
                track.update_no_measurement(current_frequency)

                # Keep track of the missed measurements of this object
                track.skipped_frames += 1


        # Delete tracks if skipped_frames too large
        self.tracks = [track for track in self.tracks if not track.skipped_frames > self.max_frame_skipped]

        

        # Predict next position for each track
        [track.KF.predict() for track in self.tracks]
        
        # Update traces for visualization
        for track in self.tracks:            
            
            track.trace.append(np.array(track.KF.state))



def createimage(w,h):
	img = np.ones((w,h,3),np.uint8)*255
	return img

