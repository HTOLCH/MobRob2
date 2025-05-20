import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import cv2
import numpy as np
import time
import threading
from ultralytics import YOLO
import os
 
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
 
        # subscription for manual activation of cone centering
        self.create_subscription(Bool, '/start_tracking', self.manual_tracking_callback, 10)
 
        self.cmd_vel_pub = self.create_publisher(Twist,'/cmd_vel', 10)
 
        self.get_logger().info("Color Detection Node 2 has started.")
        self.window_created = False
 
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "best.pt")
 
        # Initialize model
        self.model = YOLO(model_path)
 
        self.last_frame_time = time.time()
 
        # for turning on the spot
        self.tracking_timer = None
        self.tracking_active = False
        self.latest_bb = None
        self.latest_image = None
 
 
    def image_callback(self, msg):
        # Use a separate thread to process the image
        # threading.Thread(target=self.process_image, args=(msg,)).start()
        self.process_image(msg)
 
 
    # for testing purposes
    # ros2 topic pub /start_tracking std_msgs/Bool "data: true"
    def manual_tracking_callback(self, msg):  
        if msg.data:
            self.get_logger().info("Manual start of cone tracking.")
            self.start_tracking_timer()
 
 
    def process_image(self, msg):
 
        # MORE PROCESSING TO REDUCE CPU???
 
        # Convert the ROS image message to an OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
 
        # Resize the image to reduce the computational load (e.g., 640x480)
        cv_image_resized = cv2.resize(cv_image, (640, 480))  # Reduce resolution
 
        # Call the function to detect cones (objects)
        self.detect_cones(cv_image_resized)
 
        # Optionally save images to a folder (uncomment the line below if needed)
        # self.save_image(cv_image_resized)
 
    def detect_cones(self, cv_image):
        # Log the time before inference for performance tracking
        start_time = time.time()
 
        # Perform object detection using Roboflow
        results = self.model(cv_image)  # returns a list
        result = results[0]            
 
        # Log the inference time
        inference_time = time.time() - start_time
        # self.get_logger().info(f"Inference time: {inference_time:.2f} seconds")
 
        # Process and save each result
        image = cv_image.copy()  # Create a copy of the image for drawing
 
        # if result:
        #     self.get_logger().info("Cone detected!!")
       
        # else:
        #     self.get_logger().info("No Cone detected...")
 
 
        for box in result.boxes:
 
            conf = float(box.conf[0])  # confidence
            if conf >= 0.80:
 
                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
 
                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 10)
 
                # Add label
                label = f"CONE {conf:.2f}"
                cv2.putText(image, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 255, 0), 4)
               
                 # Calculate and draw center point
                center_x = int((x1 + x2) // 2)
                center_y = int((y1 + y2) // 2)
                cv2.circle(image, (center_x, center_y), radius=10, color=(255, 0, 0), thickness=-1)  # filled blue circle
 
                # Display center coordinates near the point
                coord_text = f"({center_x}, {center_y})"
                cv2.putText(image, coord_text, (center_x + 10, center_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
 
 
                # save the latest bounding box and image
                self.latest_bb = (x1, y1, x2, y2)
                self.latest_image = image
            else:
                self.latest_bb = None
 
        # Display the annotated image
        self.display_image(image)
 
    def display_image(self, cv_image):
        # Create the window only once
        if not self.window_created:
            cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
            self.window_created = True
 
        # Resize the window
        cv2.resizeWindow('Original Image', 640, 480)
 
        # Show the image in the window
        cv2.imshow('Original Image', cv_image)
 
        # Wait for a key press to update the window and refresh it
        cv2.waitKey(1)
 
        # Optionally, close windows if 'q' is pressed (if you want to close early)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
 
    def save_image(self, cv_image):
        # Make sure the /images directory exists
        save_dir = "/workspace/saved_images"
        os.makedirs(save_dir, exist_ok=True)
 
        # Create filename with timestamp
        # timestamp = time.strftime("%Y%m%d-%H%M%S")
        image_filename = f"{save_dir}/image.jpg"
 
        # Save the image
        cv2.imwrite(image_filename, cv_image)
        self.get_logger().info(f"Image saved as: {image_filename}")
 
    def start_tracking_timer(self):
        if not self.tracking_timer:
            self.tracking_timer = self.create_timer(0.1, self.tracking_timer_callback)
            self.tracking_active = True
 
    def stop_tracking_timer(self):
        if self.tracking_timer:
            self.tracking_timer.cancel()
            self.tracking_timer = None
            self.tracking_active = False
            self.cmd_vel_pub.publish(Twist())  # stop robot
 
    def tracking_timer_callback(self):
        if not self.tracking_active:
            return
 
        if self.latest_bb is None:
            self.get_logger().info("no bb")
            twist = Twist()
            twist.angular.z = 0.6
            self.cmd_vel_pub.publish(twist)
            return
 
        x1, y1, x2, y2 = self.latest_bb
        center_x = int((x1 + x2) / 2)
        image_center_x = self.latest_image.shape[1] // 2
        diff_x = image_center_x - center_x
 
        if abs(diff_x) > 15:
            twist = Twist()
            twist.angular.z = 0.005 * min(diff_x,500)
            self.cmd_vel_pub.publish(twist)
        else:
            self.get_logger().info("Cone centered.")
            self.stop_tracking_timer()
 
            # save image
            self.save_image(self.latest_image)
 
 
def main(args=None):
    rclpy.init(args=args)
    node = ColorDetectionNode()
 
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
 
if __name__ == '__main__':
    main()
 