#!/usr/bin/env python3
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
import message_filters
from arrow_detection.msg import ArrowDetection2D, ArrowDetection3D
import numpy as np

class Arrow3DCalculator:
    def __init__(self):
        rospy.init_node('coordinates_3d')
        self.bridge = CvBridge()
        
        # Camera calibration
        self.camera_info = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        
        # Publishers and Subscribers
        self.coordinates_3d_pub = rospy.Publisher('/arrow_detection/coordinates_3d', ArrowDetection3D, queue_size=1)
        self.depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        self.detection_sub = message_filters.Subscriber('/arrow_detection/coordinates_2d', ArrowDetection2D)
        self.camera_info_sub = rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.camera_info_callback)
        
        # Synchronizer with larger queue size and slop time
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.depth_sub, self.detection_sub],
            queue_size=30,
            slop=0.1
        )
        self.ts.registerCallback(self.callback)
        
        rospy.loginfo("3D coordinate calculator initialized")

    def camera_info_callback(self, msg):
        if self.camera_info is None:
            self.camera_info = msg
            self.fx = msg.K[0]
            self.fy = msg.K[4]
            self.cx = msg.K[2]
            self.cy = msg.K[5]
            rospy.loginfo(f"Camera intrinsics: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")

    def get_3d_point(self, x, y, depth_image):
        if self.camera_info is None:
            return None

        try:
            height, width = depth_image.shape
            x, y = int(x), int(y)
            
            if not (0 <= x < width and 0 <= y < height):
                return None

            # Get depth value from a small window around the point
            window_size = 5
            x_start = max(0, x - window_size//2)
            x_end = min(width, x + window_size//2)
            y_start = max(0, y - window_size//2)
            y_end = min(height, y + window_size//2)
            
            depth_window = depth_image[y_start:y_end, x_start:x_end]
            depth_values = depth_window[depth_window > 0]  # Filter out zero values
            
            if len(depth_values) == 0:
                return None
                
            depth = np.median(depth_values)  # Use median for robustness
            z = depth / 1000.0  # Convert to meters
            
            # Calculate 3D coordinates
            x_3d = (x - self.cx) * z / self.fx
            y_3d = (y - self.cy) * z / self.fy
            
            return (x_3d, y_3d, z)

        except Exception as e:
            rospy.logerr(f"3D point calculation error: {str(e)}")
            return None

    def callback(self, depth_msg, detection_2d):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
            point_3d = self.get_3d_point(detection_2d.x, detection_2d.y, depth_image)
            
            if point_3d is not None:
                detection_3d = ArrowDetection3D()
                detection_3d.header = detection_2d.header
                detection_3d.label = detection_2d.label
                detection_3d.confidence = detection_2d.confidence
                detection_3d.x = point_3d[0]
                detection_3d.y = point_3d[1]
                detection_3d.z = point_3d[2]
                
                self.coordinates_3d_pub.publish(detection_3d)
                rospy.loginfo(f"3D coordinates: ({point_3d[0]:.3f}, {point_3d[1]:.3f}, {point_3d[2]:.3f})")
            else:
                rospy.logwarn("Could not calculate 3D point")

        except Exception as e:
            rospy.logerr(f"Callback error: {str(e)}")

if __name__ == '__main__':
    try:
        node = Arrow3DCalculator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
