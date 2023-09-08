#!/usr/bin/env python3
import rospy

class DetectorData():
    def __init__(self) -> None:
        self.subscriber = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.callback)

    def callback(self):
        pass

if __name__ == "__main__":
    rospy.init_node("product_pose_estimator")
    
    while True:
        print("running")
        rospy.sleep(5)