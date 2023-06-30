#!/usr/bin/env python  
import rospy
from tf.transformations import quaternion_from_euler
import tf2_ros
from geometry_msgs.msg import TransformStamped
from ahold_product_detection.msg import FloatList



def handle_detected_product(msg):
    # Convert message to a tf2 frame when message becomes available
    br = tf2_ros.TransformBroadcaster()
    t = TransformStamped()

    x, y, z, theta, phi, psi = msg.data
    
    t.header.stamp = rospy.Time.now()
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



def main():
    rospy.init_node('tf2_product_broadcaster')
    rospy.Subscriber('detected_ahold_product', FloatList, handle_detected_product)
    rospy.spin()



if __name__ == '__main__':
    main()