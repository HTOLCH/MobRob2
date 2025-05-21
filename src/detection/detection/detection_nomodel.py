import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import math
# from ultralytics import YOLO
import os
from geometry_msgs.msg import Point32
from std_msgs.msg import Float32MultiArray, String
from interfaces.srv import TakePhoto

 #  ros2 run detection detection
 #  source install/setup.bash
 # 
 
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
            1  # Queue size; adjust as needed
        )
 
        # Create a service to take a photo
        self.take_photo_service = self.create_service(
            TakePhoto,
            'take_photo',
            self.take_photo_callback
        )
 
        # Publisher for image with bounding boxes
        self.image_pub = self.create_publisher(Image, 'image_with_bboxes', 10)
 
        # Publisher for object detected
        self.object_detected = self.create_publisher(String, 'bounding_boxes_object', 10) # bbox of object
 
        self.get_logger().info("Color Detection Node has started.")
        self.window_created = False
 
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # model_path = os.path.join(script_dir, "best.pt")
 
        # for cone
        self.latest_bb = None
 
        # for object
        self.object_bb = None
 
        self.bool_detection = False
 
        # Initialize model
        # self.model = YOLO(model_path)
 
        self.last_frame_time = time.time()
 
        self.frame_counter = 0
 
        self.latest_image = None

        self.processing_image = False
           
    def image_callback(self, msg):
        #if self.frame_counter % 2 == 0:  # Skip frames
        #if not self.processing_image:
        #    self.processing_image = True
        self.process_image(msg) 
        #    self.frame_counter += 1
 
    def process_image(self, msg):
        # Convert the ROS image message to an OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
 
        self.latest_image = cv_image
 
        # Resize the image to reduce the computational load (e.g., 640x480 -> 320x240)
        cv_image_resized = cv2.resize(cv_image, (320, 240))  # Reduce resolution for inference
 
        # Call the function to detect cones (objects)
        self.find_best_shape(cv_image)  

        self.processing_image = False
   
    def find_best_shape(self, original_image, area_threshold=10000):

        image = original_image.copy() 

        bbox_coordinates = String()

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        #Red ranges 
        lower_red1 = np.array([0, 120, 70]) 
        upper_red1 = np.array([25, 255, 255]) 
        lower_red2 = np.array([170, 120, 70]) 
        upper_red2 = np.array([180, 255, 255]) 

        # Yellow range #CHNAGED HERE
        # self.get_logger().info(f"changed yellow")
        # lower_yellow = np.array([30, 100, 100]) 
        # upper_yellow = np.array([65, 255, 255])

        # lower_yellow = np.array([20, 100, 100])
        # upper_yellow = np.array([30, 255, 255])

        # orange range
        # lower_orange = np.array([20, 120, 120])
        # upper_orange = np.array([35, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        # mask3 = cv2.inRange(hsv, lower_yellow, upper_yellow)
        # mask4 = cv2.inRange(hsv, lower_orange, upper_orange)
        # mask = cv2.bitwise_or(mask1, mask2, mask3, mask4)
        mask = mask1 | mask2 
        
        mask_blur = cv2.GaussianBlur(mask, (5, 5), 0)
        kernel = np.ones((7, 7), np.uint8)
        mask_clean = cv2.morphologyEx(mask_blur, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

        # Contours
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = -1
        biggest_box = None

        # self.get_logger().info(f"Number of contours: {len(contours)}")

        for cnt in contours:
            epsilon = 0.0005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            area = cv2.contourArea(approx)

            # self.get_logger().info(f"Area of contour: {area}")

            if area > area_threshold:

                x, y, w, h = cv2.boundingRect(approx)
                x1, y1 = x, y
                x2, y2 = x + w, y + h
                bbox = (x1, y1, x2, y2)

                # figure out aspect ratio of buckets and cone
                aspect_ratio = w / float(h)

                # plot contour and aspect ration in purple
                # AREA IS CONTOUR AREA!!!
                label = f"AR: {aspect_ratio:.2f}, A: {area:.1f}"
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(image, label, (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (255, 0, 255), 4)

                # Check if it's square-ish or upright rectangle - square bucket
                if (0.8 <= aspect_ratio <= 1.2) and (50000 <= area <= 120000): 
                    self.get_logger().info(f"Probably a small bucket")
                    if area > max_area:
                        max_area = area
                        biggest_box = approx

                elif (0.7 <= aspect_ratio <= 0.9) and (110000 <= area):
                    self.get_logger().info(f"Probably the big red bucket")
                    if area > max_area:
                        max_area = area
                        biggest_box = approx


        if max_area > 0:

            # detected an object
            self.get_logger().info(f"Found an object")
            self.object_bb = biggest_box

            # make bb green
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)
            label = f"BOBJECT"
            cv2.putText(image, label, (x1, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (0, 255, 0), 4)

        else:

            # no objects detected
            self.get_logger().info(f"No objects detected")
            self.object_bb = None
               
        # Check if image is a valid NumPy array with at least 2 dimensions (height, width)
        if not isinstance(image, np.ndarray) or image.ndim < 2:
            self.get_logger().info("[WARNING] Invalid image format. Skipping publish.")
            image = original_image.copy()

        try:
            self.publish_image(image)  # Try to publish the image
        except Exception as e:
            self.get_logger().info(f"[ERROR] Failed to publish image: {e}")
            image = original_image.copy()
            self.publish_image(image)
            
 
        # Publish bounding box coordinates
        if self.object_bb is not None:
            bbox_coordinates.data = str(list(self.object_bb) + [None])

        self.object_detected.publish(bbox_coordinates)
 
 
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