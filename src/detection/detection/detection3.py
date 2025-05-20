import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from ultralytics import YOLO
import os
from geometry_msgs.msg import Point32
from std_msgs.msg import Float32MultiArray
from interfaces.srv import TakePhoto


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

        # Create a service to take a photo
        self.take_photo_service = self.create_service(
            TakePhoto, 
            'take_photo', 
            self.take_photo_callback
        )

        # Publisher for image with bounding boxes
        self.image_pub = self.create_publisher(Image, 'image_with_bboxes', 10)

        # Publisher for bounding box coordinates
        self.bbox_pub = self.create_publisher(Float32MultiArray, 'bounding_boxes', 10)

        self.get_logger().info("Color Detection Node has started.")
        self.window_created = False

        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "best.pt")

        self.latest_bb = None

        # Initialize model
        self.model = YOLO(model_path)

        self.last_frame_time = time.time()

        self.frame_counter = 0

        self.latest_image = None
        

    def image_callback(self, msg):
        if self.frame_counter % 5 == 0:  # Skip frames
            self.process_image(msg)
        self.frame_counter += 1

    def process_image(self, msg):
        # Convert the ROS image message to an OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        self.latest_image = cv_image

        # Resize the image to reduce the computational load (e.g., 640x480 -> 320x240)
        cv_image_resized = cv2.resize(cv_image, (320, 240))  # Reduce resolution for inference

        # Call the function to detect cones (objects)
        self.detect_cones(cv_image, cv_image_resized)  # Pass both original and resized images

    def detect_cones(self, original_image, cv_image_resized):
        # Perform object detection using YOLO
        results = self.model(cv_image_resized)  # returns a list
        result = results[0]

        image = original_image.copy()  # Create a copy of the original image for drawing

        bbox_coordinates = Float32MultiArray()

        found = False

        for box in result.boxes:
            conf = float(box.conf[0])  # confidence
            if conf >= 0.80:
                # Bounding box coordinates for resized image
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Scale bounding box coordinates back to the original resolution
                x1, y1, x2, y2 = int(x1 * (original_image.shape[1] / 320)), int(y1 * (original_image.shape[0] / 240)), \
                                   int(x2 * (original_image.shape[1] / 320)), int(y2 * (original_image.shape[0] / 240))

                
                self.get_logger().info(f"Bounding box coordinates: {x1}, {y1}, {x2}, {y2}")

                # Draw bounding box on the original image
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

                # Save the latest bounding box and image
                self.latest_bb = (x1, y1, x2, y2)

                found = True
                
        if not found:
            self.latest_bb = None

        # Publish the image with bounding boxes
        self.publish_image(image)

        # Publish bounding box coordinates
        if self.latest_bb is not None:
            bbox_coordinates.data = list(self.latest_bb)

        self.get_logger().info(f"Bounding box coordinates: {self.latest_bb}")
        self.bbox_pub.publish(bbox_coordinates)

    def publish_image(self, image):
        # Convert OpenCV image to ROS Image message
        ros_image = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")

        # Publish the image
        self.image_pub.publish(ros_image)

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

    def take_photo_callback(self, request, response):
        try:
            # Action to take a photo
            self.get_logger().info("Taking photo...")
            self.save_image()

            # Set response fields
            response.success = True
            response.message = "Photo taken and saved successfully."
            
            self.get_logger().info("Photo taken successfully.")
        except Exception as e:
            # In case something goes wrong, handle the exception
            response.success = False
            response.message = f"Failed to take photo: {str(e)}"
            self.get_logger().warn(response.message)

        return response

    def save_image(self):
        # Save image to a folder for debugging or logging
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        image_filename = f"/workspace/saved_images/{timestamp}_image.jpg"  # Adjust path as needed
        cv2.imwrite(image_filename, self.latest_image)
        self.get_logger().info(f"Image saved as: {image_filename}")

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
