import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import threading
import time
from ultralytics import YOLO
import os
from geometry_msgs.msg import Point32
from std_msgs.msg import Float32MultiArray  # For bounding box coordinates
from geometry_msgs.msg import Twist

class ColorDetectionNode(Node):
    def __init__(self):
        super().__init__('color_detection_node')

        # Initialize CvBridge to convert ROS images to OpenCV images
        self.bridge = CvBridge()

        # Create a subscriber to the camera topic
        self.subscription = self.create_subscription(
            Image,
            '/oak/rgb/image_raw',
            self.image_callback,
            10  # Queue size; adjust as needed
        )

        # Publisher for image with bounding boxes
        self.image_pub = self.create_publisher(Image, 'image_with_bboxes', 10)

        # Publisher for bounding box coordinates
        self.bbox_pub = self.create_publisher(Float32MultiArray, 'bounding_boxes', 10)

        self.get_logger().info("Color Detection Node has started.")
        
        # Create a thread-safe queue for image processing
        self.image_queue = []
        self.image_lock = threading.Lock()

        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "best.pt")

        self.latest_bb = None
        self.model = YOLO(model_path)

        # Start image processing thread
        self.processing_thread = threading.Thread(target=self.process_images)
        self.processing_thread.daemon = True  # Ensure thread exits when main program exits
        self.processing_thread.start()

    def image_callback(self, msg):
        # Add incoming image to the queue for processing
        with self.image_lock:
            self.image_queue.append(msg)

    def process_images(self):
        while rclpy.ok():
            # Process image queue in the background
            if self.image_queue:
                # Take the first image from the queue
                with self.image_lock:
                    msg = self.image_queue.pop(0)
                self.process_image(msg)
            time.sleep(0.01)  # Prevent busy-waiting

    def process_image(self, msg):
        # Convert the ROS image message to an OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Resize the image to reduce computational load (e.g., 320x240)
        cv_image_resized = cv2.resize(cv_image, (320, 240))

        # Perform object detection using the YOLO model
        self.detect_cones(cv_image_resized)

    def detect_cones(self, cv_image):
        # Perform object detection
        results = self.model(cv_image)  # Returns a list
        result = results[0]

        # Process and save each result
        image = cv_image.copy()  # Create a copy of the image for drawing
        bbox_coordinates = Float32MultiArray()

        for box in result.boxes:
            conf = float(box.conf[0])  # Confidence
            if conf >= 0.80:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 10)
                label = f"CONE {conf:.2f}"
                cv2.putText(image, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

                # Publish bounding box coordinates
                self.latest_bb = (x1, y1, x2, y2)
            else:
                self.latest_bb = None

        # Publish the image and bounding box coordinates in the main thread
        self.publish_image(image)
        if self.latest_bb is not None:
            bbox_coordinates.data = list(self.latest_bb)
            self.bbox_pub.publish(bbox_coordinates)

    def publish_image(self, image):
        # Convert OpenCV image to ROS Image message
        ros_image = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")

        # Publish the image
        self.image_pub.publish(ros_image)

def main(args=None):
    rclpy.init(args=args)
    node = ColorDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.processing_thread.join()  # Ensure the processing thread exits
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
