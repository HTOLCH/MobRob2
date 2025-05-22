import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from visualization_msgs.msg import Marker
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from tf_transformations import quaternion_from_euler
from math import radians
import time
from std_msgs.msg import String
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import math
import cv2
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64
from interfaces.srv import TakePhoto
from time import sleep
from tf2_ros import TransformException, Buffer, TransformListener
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import std_msgs.msg
import tf2_geometry_msgs
import tf2_ros
import random
import math
import threading
import ast
import itertools
from nav_msgs.msg import Path
from std_msgs.msg import Header
import string

class AutoNavigator(Node):
    def __init__(self):
        super().__init__('auto_navigator')
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Create a client for the 'take_photo' service
        self.camera_client = self.create_client(TakePhoto, 'take_photo')

        self.distances_publisher = self.create_publisher(String, 'distances', 10)
        self.twist_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_pub = self.create_publisher(Marker, '/visualization_marker', 10)
        self.activate_detection = self.create_publisher(Bool, 'enable_detection', 10)
        self.path_pub = self.create_publisher(Path, '/plan', 10)  

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.current_twist = Twist()

        self.bounding_box_detected = False

        #Set up bounding box parameters received
        self.bb_x1 = None
        self.bb_x2 = None
        self.bb_y1 = None
        self.bb_y2 = None
        self.bb_item = None

        self.heading = 0.0

        #Bounding box subscription
        self.subscription = self.create_subscription(
            String,  # Replace with the actual bounding box message type
            'bounding_boxes_object',
            self.bounding_box_callback,
            10)
        
        # Subscriber to the 'scan' topic
        self.scan_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Subscriber to the 'phidget' topic
        self.phidget_subscription = self.create_subscription(
            Float64,
            'phidget',
            self.phidget_callback,
            10
        )

        # Subscriber to the 'auto_status' topic
        self.auto_status_subscription = self.create_subscription(
            Bool,
            'auto_status',
            self.auto_status_callback,
            10
        )

        self.skip_mapping_subsciption = self.create_subscription(
            Bool,
            'skip_mapping',
            self.skip_exploration_callback,
            10
        )

        self.latest_odom = None
        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)

        self.min_forward_float = None

        self.skip_exploration = False

        #List for testing
        self.object_list=[
            ("1", -2, -5, 0, "f"),
            ("2", -2, 2, 0, "f"),
            ("3", 3, 3, 0, "f"),
            ("4", 4, -4, 0, "f"),
            ("5", 5, 5, 0, "f"),
            ("6", -7, -7, 0, "f"),
            ("7", -7, -7, 0, "f"),
            ("r", -2, -2, 0, "f"),
            ("y", -5, -2, 0, "f")
        ]

        self.activated = False

        self.x_location = None
        self.y_location = None

        self.first_stop = True


    def auto_status_callback(self,msg):
        if msg.data == True:
            self.activated = True
        else:
            self.activated = False

        #self.get_logger().info(str(self.activated))

    def skip_exploration_callback(self, msg):
        self.skip_exploration = msg.data
    
    def odom_callback(self, msg):
        self.latest_odom = msg

    def phidget_callback(self,msg):
        self.heading = msg.data

    def publish_goal_marker(self, x, y, idx, colour, delete=False):

        if colour == "g":
            g = 1.0
            r = 0.0
        else:
            g = 0.0
            r = 1.0

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "goal_markers"
        marker.id = idx
        marker.type = Marker.SPHERE
        marker.action = Marker.DELETE if delete else Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3

        marker.color.a = 1.0  # Alpha
        marker.color.r = r
        marker.color.g = g
        marker.color.b = 0.0

        self.marker_pub.publish(marker)


    def scan_callback(self, msg):

        def valid_ranges(data):
            return [r for r in data if r > 0.2 and r < 15]
        
        #640 samples in sim,
        #1440 samples on richbeam lidar
        lidar_samples = 1440

        lower_right = int(lidar_samples/4.5)
        upper_right = int(lidar_samples/2.3)

        lower_forward = int(upper_right)
        upper_forward = int(lidar_samples/1.8)

        lower_left = int(upper_forward)
        upper_left = int(lidar_samples/1.286)

        try:
            regions = {
                'right': msg.ranges[lower_right:upper_right],
                'forward': msg.ranges[lower_forward:upper_forward],
                'left': msg.ranges[lower_left:upper_left]
            }

            #self.get_logger().info(f"Ranges length: {len(msg.ranges)}")
            

            filtered_right = valid_ranges(regions['right'])
            filtered_forward = valid_ranges(regions['forward'])
            filtered_left = valid_ranges(regions['left'])

            # Find minimum in each region
            if filtered_right:
                self.min_right = f"{min(filtered_right):.2f}"
            else:
                self.min_right = "Out of range"

            if filtered_forward:

                self.min_forward_float = min(filtered_forward)
                self.min_forward = f"{min(filtered_forward):.2f}"
            else:
                self.min_forward = "Out of range"
    
            if filtered_left:
                self.min_left = f"{min(filtered_left):.2f}"
            else:
                self.min_left = "Out of range"

        except ValueError as e:
            self.get_logger().error(f"Error processing ranges: {e}")
            return


        self.distances = f"Left: {self.min_left} | Forward: {self.min_forward} | Right: {self.min_right}"
        # Publish the distances
        message = String()
        message.data = self.distances
        self.distances_publisher.publish(message)


    def bounding_box_callback(self, msg):
        self.get_logger().info("Bounding box detected. Interrupting current path.")
        
        #Bounding boxes will be published as a list: [x1,x2,y1,y2,item]
        data = list(msg.data)

        self.bb_x1 = data[0]
        self.bb_x2 = data[1]
        self.bb_y1 = data[2]
        self.bb_y2 = data[3]
        self.bb_item = data[4]

        object_in_list = False

        #Only set true if the number/ object detected has not already been stored:
        for obj in self.object_list:
            if obj[0] == str(self.bb_item) and obj[4] == "n": #Found the same object that is found during test
                object_in_list = True
                self.get_logger().info("Detected object has already been stored, ignoring.")         
        
        if not object_in_list:
            self.get_logger().info("Detected object has not been stored, pausing mapping...")
            self.bounding_box_detected = True
        
    def clear_path(self):
        empty_path = Path()
        empty_path.header = Header()
        empty_path.header.stamp = self.get_clock().now().to_msg()
        empty_path.header.frame_id = "map"  # Use correct frame

        self.path_pub.publish(empty_path)

    def send_goal(self, x, y, theta_deg):
        self.get_logger().info(f"Sending goal to x={x}, y={y}, theta={theta_deg}°")

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        # Position
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y

        # Orientation
        q = quaternion_from_euler(0, 0, radians(theta_deg))
        goal_msg.pose.pose.orientation.x = q[0]
        goal_msg.pose.pose.orientation.y = q[1]
        goal_msg.pose.pose.orientation.z = q[2]
        goal_msg.pose.pose.orientation.w = q[3]

        self._action_client.wait_for_server()
        return self._action_client.send_goal_async(goal_msg)

    def handle_bounding_box(self):

        # Take photo
        self.take_photo()

        #Record object location
        self.get_logger().info(f"Recording object position")   

        #Call marker function
        self.publish_goal_marker(self.x_location, self.y_location, self.bb_item, "r", delete=False)

        # Store in list
 
        for i, obj in enumerate(self.object_list):
            if obj[0] == str(self.bb_item):  # Only override if we find the same item and it's not a failsafe in the list.
                # Override the existing values
                self.object_list[i] = (self.bb_item, self.x_location, self.y_location, 0, "n")
                break  # Stop once the item is updated

        #Print new item list
        self.get_logger().info(f"Object list:\n{self.object_list}")

        # Resume normal path
        self.bounding_box_detected = False

        return

    def euclidean(self,p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def total_path_distance(self,path):
        return sum(self.euclidean(path[i], path[i+1]) for i in range(len(path) - 1))
    
    def find_shortest_path(self, points, start=None):
        if start:
            points = [start] + [p for p in points if p != start]
        shortest = None
        min_dist = float('inf')

        for perm in itertools.permutations(points[1:]):
            path = [points[0]] + list(perm)
            dist = self.total_path_distance(path)
            if dist < min_dist:
                min_dist = dist
                shortest = path

        return shortest

    def explore(self):

        self.get_logger().info("Exploring")

        #Call a while loop for 20 minutes
        start_time = time.time()
        self.outside_bounds = False

        while (time.time() - start_time) < (20 * 60):  # 20 minutes
            rclpy.spin_once(self, timeout_sec=2)

            #sleep(1)

            while self.activated == False:
                rclpy.spin_once(self, timeout_sec=0.1)

            #Check to see if skip button has been pressed
            if self.skip_exploration:
                break
            
            #Start moving forwards
            self.current_twist.linear.x = 1.0
            self.current_twist.angular.z = 0.0
            self.twist_pub.publish(self.current_twist)

            #Get the current map location
            try:
                trans = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
                #self.get_logger().info(f"Map Pose: {trans.transform.translation}, {trans.transform.rotation}")

                #Extract x and y coordinates
                x = trans.transform.translation.x
                y = trans.transform.translation.y

                self.x_location = x
                self.y_location = y

            except tf2_ros.TransformException as e:
                self.get_logger().error(f"Transform lookup failed: {str(e)}")
                continue

            self.get_logger().info(f"X: {x}")
            self.get_logger().info(f"Y: {y}")  

            #If we reach the edge of the map, or find an object, turn to an angle behind the pioneer.

            self.get_logger().info(f"Outside bounds?: {self.outside_bounds}")

            x_min = -4.0
            x_max = 6.0
            y_min = -6.0
            y_max = 6.0

            #Mode switch gap
            msg = 0.1

            if x > x_max and not self.outside_bounds:
                self.get_logger().warn("Outside bounds Adjusting...")
                self.outside_bounds = True  # Set flag

                #Obtain desired angle
                desired_angle = random.uniform(135, 225)
                #Turn to this angle
                self.turn_to_angle(desired_angle)
            
            if x < x_min and not self.outside_bounds:
                self.get_logger().warn("Outside bounds! Adjusting...")
                self.outside_bounds = True  # Set flag

                #Obtain desired angle
                desired_angle = random.uniform(-45, 45)
                #Turn to this angle
                self.turn_to_angle(desired_angle)
            
            if y > y_max and not self.outside_bounds:
                self.get_logger().warn("Outside bounds! Adjusting...")
                self.outside_bounds = True  # Set flag

                #Obtain desired angle
                desired_angle = random.uniform(45, 135)
                #Turn to this angle
                self.turn_to_angle(desired_angle)
            
            if y < y_min and not self.outside_bounds:
                self.get_logger().warn("Outside bounds! Adjusting...")
                self.outside_bounds = True  # Set flag

                #Obtain desired angle
                desired_angle = random.uniform(225, 315)
                #Turn to this angle
                self.turn_to_angle(desired_angle)

            if y < (y_max - msg) and y > (y_min + msg) and x < (x_max - msg) and x > (x_min + msg):
                #We are back within bounds, allow outside bounds functions to run again.
                self.outside_bounds = False

            #If the distance now is < 0.7, trigger an estop.
            if float(self.min_forward_float) < 0.7:
                #Stop the robot.
                self.current_twist.linear.x = 0.0
                self.current_twist.angular.z = 0.0
                self.twist_pub.publish(self.current_twist)

                #Call rosbag to save published data.
                

                #Keep in estop while loop until auto mode turns to false (deadman triggered)
                while self.activated == True:
                    rclpy.spin_once(self, timeout_sec=0.1)

                #auto mode has switched off, resume...

            #Check lidar distance 
            if float(self.min_forward_float) < 1.4:
                #We've found a potential object of interest, stop the robot and enable detection
                self.get_logger().info("Something in front")
                self.current_twist.linear.x = 0.0
                self.current_twist.angular.z = 0.0
                self.twist_pub.publish(self.current_twist)

                #Turn on detection
                msg = Bool()
                msg.data = True
                self.activate_detection.publish(msg)

                sleep(1)

                # Wait for bounding box response
                detection_timeout = 8  # Wait time (seconds)
                start_time = time.time()

                bounding_box_received = False  # Track detection status

                while (time.time() - start_time) < detection_timeout:
                    rclpy.spin_once(self, timeout_sec=0.1)  # Process incoming messages
                    if self.bounding_box_detected: 
                        bounding_box_received = True
                        break

                #If we got a bounding box, call handle_bounding_box()
                if bounding_box_received:
                    self.get_logger().info("Object detected! Handling bounding box...")
                    self.handle_bounding_box()  # Call the handler function
                
                else:
                    self.get_logger().info("Did not detect an object.")

                #Turn off detection
                msg = Bool()
                msg.data = False
                self.activate_detection.publish(msg)

                #Regardless, turn away and continue exploration
                #Choose whether to go left or right
                direction = random.uniform(1,2)

                if direction == 1:
                    desired_angle = random.uniform(self.heading + 90, self.heading + 160)
                else:
                    desired_angle = random.uniform(self.heading - 90, self.heading - 160)

                #Turn to this angle
                self.turn_to_angle(desired_angle)


        self.get_logger().info("Exploration complete")
        #sleep(2)   
        
        if self.first_stop:
            #Mapping finished, stop
            self.first_stop = False
            self.current_twist.linear.x = 0.0
            self.current_twist.angular.z = 0.0
            self.twist_pub.publish(self.current_twist)
        

        #Call the go to points function
        self.drive_to_points()

    
    def drive_to_points(self):
        self.get_logger().info("Driving to points")

        drive_targets = []
        home_position = (0,0,0)

        if not self.object_list:
            self.get_logger().warn("No objects available to drive to.")
            return

        while True:
            points_str = input("Enter object indices (e.g. 1,2,5): ")
            try:
                indices = ast.literal_eval(f"[{points_str}]")
                if not all(isinstance(i, int) for i in indices):
                    raise ValueError("All inputs must be integers.")

                selected_coords = []
                for idx in indices:
                    if 0 <= idx < len(self.object_list):
                        obj = self.object_list[idx-1]
                        selected_coords.append((str(idx), obj[1], obj[2], obj[3]))  # label, x, y, phi
                    else:
                        raise ValueError(f"Index {idx} out of range.")

                break  # Input is valid, exit loop

            except Exception as e:
                self.get_logger().error(f"Invalid input: {e}")
                print("Please try again.\n")

        for label, x, y, phi in selected_coords:
            self.get_logger().info(f"Driving to object {label} at coordinates x={x}, y={y}, phi={phi}")
            drive_targets.append((label, x, y, phi))
        
        self.get_logger().info(f"targets list: {drive_targets}")
        #Find the fastest path between all the targets
        positions = [(x, y, phi) for (_, x, y, phi) in drive_targets]

        fastest_path = self.find_shortest_path(positions, start=(0,0,0))

        fastest_path.append(home_position)

        self.get_logger().info(f"Fastest path: {fastest_path}")

        self.get_logger().info(f"Navigating path...")

        #Start at 1
        route_index = 0
        completed_exploration_waypoint_rerun = False

        while not completed_exploration_waypoint_rerun:
            #sleep(1)

            while self.activated == False:
                rclpy.spin_once(self, timeout_sec=0.1)

            while route_index < len(fastest_path) and self.activated:
                    x, y, theta = fastest_path[route_index]
                    x = float(x)
                    y = float(y)
                    theta = float(theta)
                    self.get_logger().info(f"Driving to object {route_index} at coordinates x={x}, y={y}, phi={theta}")
                    self.publish_goal_marker(x, y, route_index, "g", delete=False)

                    future = self.send_goal(x, y, theta)
                    rclpy.spin_until_future_complete(self, future)
                    goal_handle = future.result()

                    if not goal_handle.accepted:
                        self.get_logger().warn(f"Goal #{route_index} rejected")
                        self.publish_goal_marker(x, y, route_index, "g", delete=True)
                        route_index += 1
                        continue

                    self.get_logger().info(f"Goal #{route_index} accepted")
                    result_future = goal_handle.get_result_async()
                    start_time = time.time()

                    while rclpy.ok() and not result_future.done():
                        rclpy.spin_once(self, timeout_sec=0.1)

                        if not self.activated:
                            self.get_logger().info("Exploration deactivated. Saving current index.")
                            goal_handle.cancel_goal_async()
                            route_index+=1
                            break


                        if time.time() - start_time > 100.0:
                            self.get_logger().warn(f"Goal #{route_index} timed out.")
                            goal_handle.cancel_goal_async()
                            route_index += 1
                            self.clear_path()
                            break

                    if result_future.done():
                        result = result_future.result().result
                        if result.error_code == 0:
                            self.get_logger().info(f"Goal #{route_index} succeeded")
                            route_index += 1
                            self.publish_goal_marker(x, y, route_index-1, "g", delete=True)

                        elif result.error_code == 102:
                            self.get_logger().warn(f"Error 102")
                            self.clear_path()
                            self.publish_goal_marker(x, y, route_index-1, "g", delete=True)

                        else:
                            self.get_logger().warn(
                                f"Goal #{route_index} failed - Code: {result.error_code}"
                            )
                            self.clear_path()
                            self.publish_goal_marker(x, y, route_index, "g", delete=True)
                    else:
                        self.get_logger().info(f"Goal #{route_index} not completed.")
                        self.clear_path()
                        self.publish_goal_marker(x, y, route_index, "g", delete=True)

            #Check to see if we have completed exploration
            if route_index == (len(fastest_path)):
                self.completed_exploration = True
                break

        self.get_logger().info("All operations completed.")
    

    def turn_to_angle(self, angle):          
        while True:  # Continuously update heading during turning
            rclpy.spin_once(self, timeout_sec=0.1)  # Frequent updates for real-time TF processing

            #Exit if not in auto mode
            if self.activated == False:
                return

            heading_rad = math.radians(self.heading)
            desired_heading_rad = math.radians(angle)
            heading_error_rad = self.normalize_angle(desired_heading_rad - heading_rad)

            self.get_logger().info(f"Current Heading: {math.degrees(heading_rad)}")
            self.get_logger().info(f"Desired Heading: {math.degrees(desired_heading_rad)}")
            self.get_logger().info(f"Heading error: {heading_error_rad}")

            if abs(heading_error_rad) < 0.2:
                return  # Stop turning once error is small enough

            # Decide turning direction
            self.current_twist.angular.z = 0.5 if heading_error_rad < 0 else -0.5
            self.current_twist.linear.x = -0.3

            self.twist_pub.publish(self.current_twist)
            #sleep(0.05)  # Minimal sleep to allow fast adjustments  

            

    def normalize_angle(self, angle):
        # Normalize angle to be within [-pi, pi]
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
    
    def take_photo(self):
        # Create a request (empty if no parameters are required)
        request = TakePhoto.Request()

        # Call the service asynchronously and handle the result
        future = self.camera_client.call_async(request)
        future.add_done_callback(self.handle_take_photo_response)

    def handle_take_photo_response(self, future):
        response = future.result()
        if response.success:
            self.get_logger().info(f"Photo taken successfully: {response.message}")
        else:
            self.get_logger().warn(f"Failed to take photo: {response.message}")


def main():
    rclpy.init()
    node = AutoNavigator()

    # Start exploration in a separate thread
    explore_thread = threading.Thread(target=node.explore)
    explore_thread.start() 

    # Use MultiThreadedExecutor to handle subscriptions
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()  # Keeps subscriptions running   

    explore_thread.join()  # Ensure exploration completes before shutdown
    node.destroy_node()
    rclpy.shutdown()



if __name__ == '__main__':
    main()
