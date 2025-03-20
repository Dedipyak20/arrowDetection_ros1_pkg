#!/usr/bin/env python3

import rospy
import cv2
import torch
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from pathlib import Path
from std_msgs.msg import Header
import sys
from arrow_detection.msg import ArrowDetection2D

# Add YOLOv5 to path
YOLOV5_PATH = "/home/external_repos/yolov5"
if YOLOV5_PATH not in sys.path:
    sys.path.append(YOLOV5_PATH)

from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device

class ArrowDetection2DNode:
    def __init__(self):
        rospy.init_node('coordinates_2d')
       
        # Initialize parameters
        self.weights = rospy.get_param('~weights_path', '/home/external_repos/yolov5/bestweight.pt')
        self.img_size = rospy.get_param('~img_size', 640)
        self.conf_thres = rospy.get_param('~conf_thres', 0.25)
        self.iou_thres = rospy.get_param('~iou_thres', 0.45)
        self.device = select_device(rospy.get_param('~device', ''))
       
        # Initialize YOLOv5 model
        self.model = DetectMultiBackend(self.weights, device=self.device)
        self.stride = self.model.stride
        self.names = self.model.names
        self.pt = self.model.pt
        self.img_size = check_img_size(self.img_size, s=self.stride)
       
        # Initialize CV Bridge
        self.bridge = CvBridge()
       
        # Publishers
        self.detection_pub = rospy.Publisher('/arrow_detection/detections', Image, queue_size=1)
        self.coordinates_pub = rospy.Publisher('/arrow_detection/coordinates_2d', ArrowDetection2D, queue_size=1)
       
        # Subscribers
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
       
        rospy.loginfo("2D Arrow detection node initialized")

    def preprocess_image(self, img):
        """Preprocess image for YOLOv5"""
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        return img

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
           
            # Prepare image for YOLOv5
            img = color_image.transpose((2, 0, 1))  # HWC to CHW
            img = self.preprocess_image(img)
           
            # Inference
            pred = self.model(img, augment=False, visualize=False)
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None,
                                    agnostic=False, max_det=1000)

            # Process detections
            det = pred[0]
            annotated_img = color_image.copy()
           
            if len(det):
                # Rescale boxes from img_size to image size
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], color_image.shape).round()
               
                # Process each detection
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    
                    # Calculate center point
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Create and publish 2D detection message
                    detection_2d = ArrowDetection2D()
                    detection_2d.header = Header()
                    detection_2d.header.stamp = rospy.Time.now()
                    detection_2d.header.frame_id = msg.header.frame_id
                    detection_2d.label = self.names[int(cls)]
                    detection_2d.confidence = float(conf)
                    detection_2d.x = center_x
                    detection_2d.y = center_y
                    self.coordinates_pub.publish(detection_2d)
                    
                    # Draw on image
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_img, label, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(annotated_img, (center_x, center_y), 4, (0, 255, 0), -1)
           
            # Publish annotated image
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_img, "bgr8")
            annotated_msg.header = msg.header
            self.detection_pub.publish(annotated_msg)

        except Exception as e:
            rospy.logerr(f"Error in callback: {str(e)}")

if __name__ == '__main__':
    try:
        node = ArrowDetection2DNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
