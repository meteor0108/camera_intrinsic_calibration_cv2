#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, threading
import rclpy
import cv2

from datetime import datetime
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pynput import keyboard

class ImagePairRecorder(Node):
    def __init__(self):
        super().__init__('image_pair_recorder')

        # Declare parameters
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('base_dir', '/your_data_path')

        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        base_dir = os.path.expanduser(self.get_parameter('base_dir').get_parameter_value().string_value)

        # 디렉토리 설정
        self.cam_dir = os.path.join(base_dir, 'task1')
        
        os.makedirs(self.cam_dir, exist_ok=True)
        
        self.bridge = CvBridge()
        self.image_msg = None
    
        # 각 카메라의 콜백을 따로 설정
        self.create_subscription(Image, self.image_topic, self.image_cb, 10)
      
        self.get_logger().info(f"{self.image_topic}")
        self.get_logger().info(f"저장 위치 →{self.cam_dir}")
        self.get_logger().info("터미널에서 's' 키를 누르면 저장합니다")

        threading.Thread(target=self.keyboard_listener, daemon=True).start()

    def image_cb(self, msg):
        self.image_msg = msg

    def keyboard_listener(self):
        def on_press(key):
            if key == keyboard.KeyCode.from_char('s'):
                self.save_current()
        with keyboard.Listener(on_press=on_press) as l: l.join()

    def save_current(self):
        if self.image_msg is None:
            self.get_logger().warning("이미지가 아직 수신되지 않았습니다.")
            return

        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        try:
            img = self.bridge.imgmsg_to_cv2(self.image_msg, 'bgr8')
           
            img_path = os.path.join(self.cam_dir, f'{stamp}.png')
            cv2.imwrite(img_path, img)

            self.get_logger().info(f"✓ 이미지 저장 완료 →\n  {img_path}\n")
        except Exception as e:
            self.get_logger().error(f"이미지 저장 실패: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ImagePairRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
