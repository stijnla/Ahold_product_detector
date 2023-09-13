import cv2
import numpy as np

class RotatedRect():

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



class RotatedRectCorners():

    def __init__(self, frame, corners, color, thickness) -> None:
        self.corners = corners
        
        self.frame = frame

        self.color = color
        self.thickness = 2
        print(self.corners[0])
        cv2.line(self.frame, self.corners[0], self.corners[1], self.color, self.thickness)
        cv2.line(self.frame, self.corners[1], self.corners[2], self.color, self.thickness)
        cv2.line(self.frame, self.corners[2], self.corners[3], self.color[::-1], self.thickness)
        cv2.line(self.frame, self.corners[3], self.corners[0], self.color, self.thickness)


