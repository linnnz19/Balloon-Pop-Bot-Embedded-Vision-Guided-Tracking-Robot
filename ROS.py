#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from std_msgs.msg import String
import cv2
from ultralytics import YOLO

# ==========================================
# Configuration Settings
# ==========================================
MODEL_PATH = '/home/worker1-pi/Desktop/Balloon_Popper_Final/best0606.onnx'
RED_ID = 2

CAM_W = 640
CAM_H = 360

# Physical Target Specifications
KILL_LINE_Y = 358
CENTER_X = (CAM_W / 2) + 80
STEERING_DEADZONE = 50
STEER_LEFT  = CENTER_X - STEERING_DEADZONE
STEER_RIGHT = CENTER_X + STEERING_DEADZONE

APPROACH_ZONE_Y = 240 

def flush_camera_buffer(cap, frames_to_drop=4):
    """
    Clear hardware buffer thoroughly to eliminate motion blur
    between high-velocity burst movements.
    """
    for _ in range(frames_to_drop):
        cap.read()

def run_hunter_system():
    # Initialize ROS node
    rospy.init_node('hunter_vision_node', anonymous=True)
    
    # Create ROS Publisher to send commands to '/hunter_cmd' topic
    cmd_pub = rospy.Publisher('/hunter_cmd', String, queue_size=1)
    rate = rospy.Rate(10) # Default loop rate when no commands are issued

    print("[INFO] Loading YOLO model (Combat Mode - Maximum Long-Range Velocity)...")
    model = YOLO(MODEL_PATH, task='detect')
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 

    rospy.loginfo("[INFO] Vision Node Ready. Publishing to /hunter_cmd")

    while not rospy.is_shutdown():
        success, frame = cap.read()
        if not success:
            continue
            
        results = model(frame, imgsz=416, conf=0.45, verbose=False)
        boxes = results[0].boxes
        
        red_balloons = []
        other_balloons = []
        
        if len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x_center = (x1 + x2) / 2
                area = (x2 - x1) * (y2 - y1)
                
                balloon_data = {'x_c': x_center, 'y_bottom': y2, 'area': area, 'id': cls_id}
                
                if cls_id == RED_ID:
                    red_balloons.append(balloon_data)
                else:
                    other_balloons.append(balloon_data)
        
        current_command = 'S'  
        sleep_time = 0.0
        target = None
        
        if red_balloons:
            target = max(red_balloons, key=lambda b: b['area'])
        elif other_balloons:
            target = max(other_balloons, key=lambda b: b['area'])
        
        if target:
            x_c = target['x_c']
            y_bottom = target['y_bottom']
            is_aligned = (STEER_LEFT < x_c < STEER_RIGHT)
            
            if y_bottom >= KILL_LINE_Y and is_aligned:
                rospy.loginfo(f"[ACTION] STRIKE! Target bottom reached: {y_bottom}")
                current_command = 'X'
                sleep_time = 1.5 
                
            elif x_c < STEER_LEFT:
                rospy.loginfo("[ACTION] Aligning Left...")
                current_command = 'L'
                sleep_time = 0.4
                
            elif x_c > STEER_RIGHT:
                rospy.loginfo("[ACTION] Aligning Right...")
                current_command = 'R'
                sleep_time = 0.4
                
            else:
                if y_bottom < APPROACH_ZONE_Y:
                    rospy.loginfo(f"[ACTION] MAXIMUM Distance Fast Approach (y2={y_bottom})")
                    current_command = 'F'
                    # Slightly increased sleep time to allow the chassis to settle
                    sleep_time = 0.6  
                else:
                    rospy.loginfo(f"[ACTION] Near Distance Micro Approach (y2={y_bottom})")
                    current_command = 'f'
                    sleep_time = 0.4
        else:
            rospy.loginfo("[ACTION] Fast Scanning Area...")
            current_command = 'S'
            sleep_time = 0.3  

        # Publish command to the ROS network
        cmd_pub.publish(current_command)

        # Handle delay and clear camera buffer
        if sleep_time > 0:
            rospy.sleep(sleep_time)
            flush_camera_buffer(cap) 
        else:
            rate.sleep()

if __name__ == '__main__':
    try:
        run_hunter_system()
    except rospy.ROSInterruptException:
        pass
