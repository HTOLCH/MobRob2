import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import math
from ultralytics import YOLO
import os
from geometry_msgs.msg import Point32
from std_msgs.msg import Float32MultiArray, String
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
            1  # Queue size; adjust as needed
        )
 
        self.detection_sub = self.create_subscription(
            String,
            'operation_state',
            self.operation_callback,
            10
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
        self.bbox_pub = self.create_publisher(Float32MultiArray, 'bounding_boxes', 10) # bbox of waypoint marker
 
        # Publisher for object detected
        self.object_detected = self.create_publisher(Float32MultiArray, 'bounding_boxes_object', 10) # bbox of object
 
        self.get_logger().info("Color Detection Node has started.")
        self.window_created = False
 
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "best.pt")
 
        # for cone
        self.latest_bb = None
 
        # for object
        self.object_bb = None
 
        self.bool_detection = False
 
        # Initialize model
        self.model = YOLO(model_path)
 
        self.last_frame_time = time.time()
 
        self.frame_counter = 0
 
        self.latest_image = None

        self.processing_image = False
 
    def operation_callback(self, msg):
 
        if msg.data == 'B': # in cones
           
            self.bool_detection  = True
 
        else:
 
            self.bool_detection = False
           
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
        self.detect_cones(cv_image, cv_image_resized, self.bool_detection)  

        self.processing_image = False
   
    def find_best_rectangle(self, image, area_threshold=500):

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        #Red ranges 
        lower_red1 = np.array([0, 120, 70]) 
        upper_red1 = np.array([10, 255, 255]) 
        lower_red2 = np.array([170, 120, 70]) 
        upper_red2 = np.array([180, 255, 255]) 


        # Yellow range #CHNAGED HERE
        lower_yellow = np.array([20, 50, 50]) 
        upper_yellow = np.array([35, 255, 255])

        # HSV range centered around orange [10, 255, 255]
        # h_ref, s_ref, v_ref = 10, 255, 255
        # lower = np.array([max(h_ref - 10, 0), max(s_ref - 90, 0), max(v_ref - 200, 0)], dtype=np.uint8)
        # upper = np.array([min(h_ref + 10, 179), min(s_ref + 90, 255), min(v_ref + 200, 255)], dtype=np.uint8)

        # lower1 = np.array([0, 100, 100], dtype=np.uint8)     # Lower bound for red
        # upper1 = np.array([10, 255, 255], dtype=np.uint8)

        # lower2 = np.array([10, 100, 100], dtype=np.uint8)    # Orange to yellow
        # upper2 = np.array([50, 255, 255], dtype=np.uint8)

        # Mask & clean
        # mask = cv2.inRange(hsv, lower, upper)

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask3 = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask = cv2.bitwise_or(mask1, mask2, mask3)
        
        mask_blur = cv2.GaussianBlur(mask, (5, 5), 0)
        kernel = np.ones((7, 7), np.uint8)
        mask_clean = cv2.morphologyEx(mask_blur, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

        # Contours
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # # Draw contours
        # cv2.drawContours(image, contours, -1, (0, 255, 0), 2)  # Green contours, thickness 2

        # # Optionally show or save the image
        # cv2.imshow("Contours", image)

        # return image, None

        max_area = 0
        best_approx = None

        for cnt in contours:
            epsilon = 0.0005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            area = cv2.contourArea(approx)

            if area > area_threshold:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)

                # Check if it's square-ish or upright rectangle
                if 0.6 <= aspect_ratio <= 1.3: # or aspect_ratio < 0.85:
                    if area > max_area:
                        max_area = area
                        best_approx = approx
                
        return best_approx, max_area

    def detect_rectangle(self, image):

        best_approx, max_area = self.find_best_rectangle(image, 5000) #CHNAGED HERE!!!

        # return_im = self.find_best_rectangle(image)
        # return return_im, [69]

        # if max_area < 1000:
        #     self.get_logger().info(f"Max area too small, redoing on bottom 2/3rds to be sure: {max_area}")
        #     height = image.shape[0]
        #     bottom_two_thirds = image[height // 3:, :]
        #     best_approx, max_area = self.find_best_rectangle(bottom_two_thirds)

        #     if best_approx is not None:
        #         # Offset points to match original image coordinates
        #         # best_approx += np.array([[0, height // 3]])
        #         # cv2.drawContours(image, [best_approx], -1, (0, 255, 0), 3)
        #         self.get_logger().info(f"Max area found after redo: {max_area}")
        #         best_approx[:, 0, 1] += height // 3  # Correct: offset Y only
        #         # cv2.drawContours(image, [best_approx], -1, (0, 255, 0), 3)
        #         # cv2.imwrite(os.path.join(output_dir, f"object_retried_{max_area}.jpeg"), image)
        #         x, y, w, h = cv2.boundingRect(best_approx)
        #         x1, y1 = x, y
        #         x2, y2 = x + w, y + h
        #         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #         bbox = (x1, y1, x2, y2)
        #         return image, bbox
            

        #     else:
        #         self.get_logger().info("No max area found after redo")
        #         return image, None

        if best_approx is not None:
            self.get_logger().info(f"Max area found: {max_area}")
            # cv2.drawContours(image, [best_approx], -1, (255, 255, 0), 3)
            # cv2.imwrite(os.path.join(output_dir, f"object_final_{max_area}.jpeg"), image)
            x, y, w, h = cv2.boundingRect(best_approx)
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            bbox = (x1, y1, x2, y2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"OBJECT"
            cv2.putText(image, label, (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (0, 255, 0), 4)
            return image, bbox
        
        else:

            self.get_logger().info(f"No max area: {max_area}")
            return image, None
    
    def is_cone_mask(self, mask):

        # Morphological closing
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # output = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        if not contours:
            self.get_logger().info("No contours found.")
            return False

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 100:
            self.get_logger().info("Contour too small.")
            return False
        
        elif area > 20000:
            self.get_logger().info("Contour too big.")
            return False

        x, y, w, h = cv2.boundingRect(largest)
        aspect_ratio = h / float(w + 1e-5)

        is_tall = aspect_ratio > 1.2
    # is_solid = solidity > 0.7
    # is_triangle_like = vertex_count <= 7
    
        if not is_tall:
            self.get_logger().info("Failed aspect ratio test")

        return is_tall #output
    
    def get_orange_score(self, image, bbox, green = False):
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_orange = np.array([0, 120, 120])
        upper_orange = np.array([21, 255, 255])
        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        orange_pixels = cv2.countNonZero(mask)

        if green:
            return orange_pixels

        is_triangle = True #self.is_cone_mask(mask) #image

        # If it has 3 vertices, likely a triangle
        if is_triangle:
            # outputpath = os.path.join("outputs", f"triangle_{orange_pixels}.jpeg")
            # cv2.imwrite(outputpath, triange)
            print("triangle found")
            return orange_pixels
        else:
            print("false detection of triangle")
            return None
        
    def get_green_score(self, image, bbox):
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        green_pixels = cv2.countNonZero(mask)

        return green_pixels

        # if green:
        #     return orange_pixels

        # is_triangle = True #self.is_cone_mask(mask) #image

        # # If it has 3 vertices, likely a triangle
        # if is_triangle:
        #     # outputpath = os.path.join("outputs", f"triangle_{orange_pixels}.jpeg")
        #     # cv2.imwrite(outputpath, triange)
        #     print("triangle found")
        #     return orange_pixels
        # else:
        #     print("false detection of triangle")
            # return Non
    
   
    def detect_cones(self, original_image, cv_image_resized, object_detection = False):
        # Perform object detection using YOLO
        results = self.model(original_image)  # returns a list
        result = results[0]
 
        image = original_image.copy()  # Create a copy of the original image for drawing
 
        bbox_coordinates = Float32MultiArray()
 
        bb_count = 0
        bbox_list = []
 
        for box in result.boxes:
            conf = float(box.conf[0])  # confidence
 
            if conf >= 0.60:
                # Bounding box coordinates for resized image
                x1, y1, x2, y2 = map(int, box.xyxy[0])
 
                # # Scale bounding box coordinates back to the original resolution
                # x1, y1, x2, y2 = int(x1 * (original_image.shape[1] / 320)), int(y1 * (original_image.shape[0] / 240)), \
                #                    int(x2 * (original_image.shape[1] / 320)), int(y2 * (original_image.shape[0] / 240))
 
                bbox = (x1, y1, x2, y2)
               
                # calculate area of the bounding box
                area = (x2 - x1) * (y2 - y1)
 
                if area > 1000 and area < 500000: # adjust based on pioneer camera
 
                    bb_count += 1
 
                    bbox_list.append([bbox,conf])

                    # Draw bounding box on the original image in purple
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 5)
 
                    # Add label
                    label = f"{bb_count} {conf:.2f} {area}"
                    cv2.putText(image, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX,
                                1.5, (0, 255, 0), 4)

                    # # Draw bounding box on the original image
                    # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 10)
 
                    # # Add label
                    # label = f"{bb_count} {conf:.2f}"
                    # cv2.putText(image, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    #             1.5, (0, 255, 0), 4)
 
                    # # Calculate and draw center point
                    # center_x = int((x1 + x2) // 2)
                    # center_y = int((y1 + y2) // 2)
                    # cv2.circle(image, (center_x, center_y), radius=10, color=(255, 0, 0), thickness=-1)  # filled blue circle
 
                    # # Display center coordinates near the point
                    # coord_text = f"({center_x}, {center_y})"
                    # cv2.putText(image, coord_text, (center_x + 10, center_y - 10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            
 
        if object_detection: # finding objects
 
            if bb_count >= 1:
 
                # orange filter
                # count orange = 1
 
                best_box = None
                worst_box = None
                min_orange = math.inf
                max_orange = -1
 
                for item in bbox_list:
                    bbox = item[0]
                    conf = item[1]
                    score = self.get_orange_score(image, bbox, True)

                    # get green score???

                    if (score is not None) and score < min_orange:
                        min_orange = score
                        worst_box = bbox

                    if (score is not None) and score > max_orange:
                        max_orange = score
                        best_box = bbox
                
                self.get_logger().info(f"cone count: {bb_count}")
                self.get_logger().info(f"orange count: {max_orange}")
                self.get_logger().info(f"green count: {min_orange}")
               
                if min_orange < 1500: # green cone detected

                    self.get_logger().info("Green node detected")

                    x1, y1, x2, y2 = worst_box 

                    # Draw green bounding box on the image
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 10)

                    # Add label
                    label = f"GREEN CONE {conf:.2f} {min_orange:.2f}"
                    cv2.putText(image, label, (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX,
                                1.5, (0, 255, 0), 4)

                    # # Calculate and draw center point
                    # center_x = int((x1 + x2) // 2)
                    # center_y = int((y1 + y2) // 2)
                    # cv2.circle(image, (center_x, center_y), radius=10, color=(255, 0, 0), thickness=-1)  # filled blue circle

                    # Display center coordinates near the point
                    self.object_bb = worst_box


                else: # orange cone detected

                    self.get_logger().info("Possible orange cone still present")
                    best_box = None
                    max_orange = -1

                    for co, item in enumerate(bbox_list):
                        bbox = item[0]
                        conf = item[1]
                        score = self.get_orange_score(image, bbox)
                        w = (bbox[2] - bbox[0])
                        l = (bbox[3] - bbox[1])
                        area =  w*l

                        # print(f"Area{co}: {area}")
                        # print(f"Orange{co} score: {score}")
                        
                        if (score is not None) and (score > max_orange) and (score < 0.7*area):
                            max_orange = score
                            best_box = bbox
                        

                    if max_orange > 1000: # orange cone detected - THRESHOLD!!!!

                        self.get_logger().info("Orange cone detected")
                
                        remove_bb = image.copy()

                        x1, y1, x2, y2 = best_box
                        # self.get_logger().info(f"{x1}, {x2}, {y1}, {y2}")
                        remove_bb[y1:, x1:x2] = (255, 255, 255)

                        self.get_logger().info("removed cone")

                        image, bbox_object = self.detect_rectangle(remove_bb)

                        # image = remove_bb
    
                        if bbox_object is not None:
                            # object found
                            self.get_logger().info("Found an object!")
                            # self.latest_bb = None                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
                            self.object_bb = bbox_object
    
    
                        else:
                            # object not found
                            self.get_logger().info("No object!")
                            self.object_bb = None

                    else:
                        # probably detected a bucket.

                        self.get_logger().info("Is this a bukcet??????")

                        orig_im = original_image.copy()

                        image, bbox_object = self.detect_rectangle(orig_im)

                        if bbox_object is not None:
                            # object found
                            self.get_logger().info("Found an object!")
                            # self.latest_bb = None
                            self.object_bb = bbox_object


                        else:
                            # object not found
                            self.get_logger().info("No object!")
                            self.object_bb = None
                       
            else: # run object detect

                orig_im = original_image.copy()

                image, bbox_object = self.detect_rectangle(orig_im)

                if bbox_object is not None:
                    # object found
                    self.get_logger().info("Found an object!")
                    # self.latest_bb = None
                    self.object_bb = bbox_object


                else:
                    # object not found
                    self.get_logger().info("Found an object!")
                    self.object_bb = None

 
        else: # Finding waypoint
 
            if bb_count >= 1:
 
                # orange filter
                # count orange = 1
 
                best_box = None
                max_orange = -1

                for co, item in enumerate(bbox_list):
                    bbox = item[0]
                    conf = item[1]
                    score = self.get_orange_score(image, bbox)
                    w = (bbox[2] - bbox[0])
                    l = (bbox[3] - bbox[1])
                    area =  w*l

                    self.get_logger().info(f"Area: {area}")
                    self.get_logger().info(f"Orange score: {score}")
                    
                    if (score is not None) and (score > max_orange) and (score < 0.65*area):
                        max_orange = score
                        best_box = bbox

                score_g = self.get_green_score(image, best_box)
                self.get_logger().info(f"Green score: {score_g}")
 
                if max_orange > 500 or score_g < 500: # orange cone detected
                    self.get_logger().info("Orange cone detected")
                    # may need to adjust threshold
                    # publish bbox to object_bbox
                    (x1, y1, x2, y2) = best_box
                    self.latest_bb = best_box

                    # Draw bounding box on the original image
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 10)
 
                    # Add label
                    label = f"CONE {bb_count} {conf:.2f} {max_orange:.2f}"
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

                else:
                    self.latest_bb = None       
           
            else:
                self.latest_bb = None
 
               
    
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
        if object_detection:
            if self.object_bb is not None:
                bbox_coordinates.data = list(self.object_bb)
 
            self.object_detected.publish(bbox_coordinates)
 
        else:
            if self.latest_bb is not None:
                bbox_coordinates.data = list(self.latest_bb)
 
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