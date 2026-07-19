#!/usr/bin/env python3
"""
cloud_vlm_node.py — 云端图生文节点 (DashScope Qwen-VL)
USB相机 -> ROS2 -> 阿里百炼API -> /vlm/description
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import requests
import base64
import time
import os


DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


class CloudVLMNode(Node):
      def __init__(self):
          super().__init__('cloud_vlm_node')

          self.declare_parameter('api_key', os.environ.get('DASHSCOPE_API_KEY', ''))
          self.declare_parameter('model', 'qwen-vl-max')
          self.declare_parameter('inference_hz', 0.1)
          self.declare_parameter('camera_topic', '/image_raw')
          self.declare_parameter('max_tokens', 300)
          self.declare_parameter('timeout', 30)
          self.declare_parameter('max_retries', 2)

          self.api_key = self.get_parameter('api_key').value
          self.model = self.get_parameter('model').value
          self.camera_topic = self.get_parameter('camera_topic').value
          self.max_tokens = self.get_parameter('max_tokens').value
          self.timeout = self.get_parameter('timeout').value
          self.max_retries = self.get_parameter('max_retries').value

          if not self.api_key:
              self.get_logger().fatal(
                  'DASHSCOPE_API_KEY 未设置。请通过环境变量或 --ros-args -p api_key:=xxx 传入'
              )
              raise RuntimeError('API key not set')

          self.bridge = CvBridge()
          self.latest_image = None
          self._busy = False

          self.sub = self.create_subscription(Image, self.camera_topic, self._img_cb, 10)
          self.pub = self.create_publisher(String, '/vlm/description', 10)

          hz = self.get_parameter('inference_hz').value
          self.timer = self.create_timer(1.0 / max(hz, 0.05), self._infer)

          self.get_logger().info(
              f'cloud_vlm ready | camera={self.camera_topic} model={self.model}'
          )

      def _img_cb(self, msg):
          self.latest_image = msg

      def _infer(self):
          if self.latest_image is None or self._busy:
              return
          self._busy = True

          try:
              import cv2
              cv_img = self.bridge.imgmsg_to_cv2(self.latest_image, 'bgr8')
              _, buf = cv2.imencode('.jpg', cv_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
              b64 = base64.b64encode(buf).decode('utf-8')

              payload = {
                  "model": self.model,
                  "messages": [{
                      "role": "user",
                      "content": [
                          {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                          {"type": "text", "text": "请用一段中文描述这张图片里的场景：有什么人物、他们在做什么、有哪些物
  体、环境怎样。"},
                      ],
                  }],
                  "max_tokens": self.max_tokens,
              }
              headers = {
                  "Authorization": f"Bearer {self.api_key}",
                  "Content-Type": "application/json",
              }

              for attempt in range(self.max_retries + 1):
                  try:
                      t0 = time.time()
                      resp = requests.post(DASHSCOPE_URL, headers=headers, json=payload, timeout=self.timeout)
                      elapsed = time.time() - t0

                      if resp.status_code == 200:
                          text = resp.json()["choices"][0]["message"]["content"]
                          self.get_logger().info(f'[VLM] {text}  ({elapsed:.1f}s)')
                          msg = String()
                          msg.data = text
                          self.pub.publish(msg)
                          break
                      else:
                          self.get_logger().warn(
                              f'API {resp.status_code} attempt {attempt+1}: {resp.text[:150]}'
                          )
                          if attempt < self.max_retries:
                              time.sleep(1)
                          else:
                              self.get_logger().error(f'API 请求最终失败: {resp.status_code}')
                  except requests.Timeout:
                      self.get_logger().warn(f'超时 attempt {attempt+1}/{self.max_retries+1}')
                  except Exception as e:
                      self.get_logger().warn(f'异常 attempt {attempt+1}: {e}')

          except Exception as e:
              self.get_logger().error(f'推理失败: {e}', throttle_duration_sec=30)
          finally:
              self._busy = False


def main(args=None):
      rclpy.init(args=args)
      try:
          rclpy.spin(CloudVLMNode())
      except RuntimeError:
          pass
      except KeyboardInterrupt:
          pass
      finally:
          rclpy.try_shutdown()


if __name__ == '__main__':
      main()
