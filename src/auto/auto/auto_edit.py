import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from visualization_msgs.msg import Marker
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from math import radians
import time
from std_msgs.msg import String
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
from std_msgs.msg import Bool
from geometry_msgs.msg import Quaternion
import ast
from nav2_simple_commander.robot_navigator import BasicNavigator
import itertools
from nav_msgs.msg import Path
from std_msgs.msg import Header

class AutoNavigator(Node):
    def __init__(self):
        super().__init__('auto_navigator')
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Create a client for the 'take_photo' service
        self.camera_client = self.create_client(TakePhoto, 'take_photo')

        self.distances_publisher = self.create_publisher(String, 'distances', 10)
        self.twist_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_pub = self.create_publisher(Marker, '/visualization_marker', 10)
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

        self.activated = False

        #List that stores the already scanned items/objects 
        self.object_list= []

        #List for testing
        self.object_list=[
            ("1", -2, -5, 0),
            ("2", -2, 2, 0),
            ("3", 3, 3, 0),
            ("4", 4, -4, 0),
            ("5", 5, 5, 0),
            ("6", -7, -7, 0),
            ("7", -7, -7, 0),
            ("r", -2, -2, 0),
            ("y", -5, -2, 0)
        ]

        self.completed_exploration = False

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

        self.received_auto_status = False

        self.latest_odom = None
        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)

        self.explore_index = 0  # Start at the first goal
        self.goal_list = []     # Store the full goal list here

        #self.navigator = BasicNavigator()
        #self.navigator.waitUntilNav2Active()

        self.get_logger().info("auto node up")

        self.min_forward = 0


        #self.timer = self.create_timer(0.5, self.explore)


    def auto_status_callback(self,msg):
        if msg.data == True:
            self.activated = True
        else:
            self.activated = False

        self.received_auto_status = True
        #self.get_logger().info(str(self.activated))
    
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

    def clear_path(self):
        empty_path = Path()
        empty_path.header = Header()
        empty_path.header.stamp = self.get_clock().now().to_msg()
        empty_path.header.frame_id = "map"  # Use correct frame

        self.path_pub.publish(empty_path)

    def scan_callback(self, msg):

        def valid_ranges(data):
            return [r for r in data if r > 0 and r < 10]
        
        #640 samples in sim,
        #1440 samples on richbeam lidar
        lidar_samples = 640

        lower_right = int(lidar_samples/4.5)
        upper_right = int(lidar_samples/2.4)

        lower_forward = int(upper_right)
        upper_forward = int(lidar_samples/1.714)

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
        #self.get_logger().info("Bounding box detected. Interrupting current path.")
        
        #Bounding boxes will be published as a list: "[x1,y1,x2,y2,item]"
        # data = ast.literal_eval(msg.data)
        # #self.get_logger().info(str(data))

        # self.bb_x1 = int(data[0])
        # self.bb_y1 = int(data[1])
        # self.bb_x2 = int(data[2])
        # self.bb_y2 = int(data[3])
        # self.bb_item = data[4]

        #Only set true if the number/ object detected has not already been stored:
        try:
            data = ast.literal_eval(msg.data)
            if not isinstance(data, list) or len(data) < 5:
                #raise ValueError("Bounding box does not have expected format")
                # Now you can safely use `data` as a list
                self.bb_x1 = int(data[0])
                self.bb_y1 = int(data[1])
                self.bb_x2 = int(data[2])
                self.bb_y2 = int(data[3])
                self.bb_item = data[4]
        except (SyntaxError, ValueError) as e:
            #self.get_logger().warn(f"Failed to parse bounding box data: {msg.data}. Error: {e}")
            return

        if any(obj[0] == self.bb_item for obj in self.object_list):
            x=1
            #print("hi")
            self.get_logger().info("Detected object has already been stored, ignoring.")    
        elif self.bb_item == None:
            x=1
        else:
            self.get_logger().info("Detected object has not been stored, pausing mapping...")
            self.bounding_box_detected = True
            

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


    def lawnmower1(self, x_min, x_max, y_min, y_max, step):
        goal_list = []
        first = True

        for y in range(y_min, y_max + 1, step):
            if (y // step) % 2 == 0:
                # Left to right, facing 0 degrees
                if first:
                    row = [(float(x), float(y), 0.0) for x in range(x_min+9, x_max + 1, step)]
                else:
                    row = [(float(x), float(y), 0.0) for x in range(x_min, x_max + 1, step)]               
            else:
                first = False
                # Right to left, facing 180 degrees
                row = [(float(x), float(y), 180.0) for x in reversed(range(x_min, x_max + 1, step))]
            goal_list += row

        return goal_list
    
    def lawnmower2(self, x_min, x_max, y_min, y_max, step):
        goal_list = []

        for x in reversed(range(x_min, x_max + 1, step)):
            if (x // step) % 2 == 0:
                row = [(float(x), float(y), 270.0) for y in reversed(range(y_min, y_max + 1, step))]               
            else:
                row = [(float(x), float(y), 90.0) for y in range(y_min, y_max + 1, step)]
            goal_list += row

        return goal_list
    
    def lawnmower3(self, x_min, x_max, y_min, y_max, step):
        goal_list = []
        count = 0

        for y in range(y_min, y_max + 1, step):
            if (y // step) % 2 == 0:
                # Left to right, facing 0 degrees
                if count == 2:
                    row = [(float(x), float(y), 0.0) for x in range(x_min, 1, step)]
                else:
                    row = [(float(x), float(y), 0.0) for x in range(x_min, x_max + 1, step)]               
            else:
                # Right to left, facing 180 degrees
                row = [(float(x), float(y), 180.0) for x in reversed(range(x_min, x_max + 1, step))]
            goal_list += row
            count += 1

        return goal_list

    def handle_bounding_box(self):
        self.get_logger().info("Detected object. Turning to face it...")

        aligned = False
        center_tolerance = 20
        # P Controller to align to object using bounding_box_center_x
        while not aligned:
            #Spin rclpy to update bounding box.
            rclpy.spin_once(self, timeout_sec=0.05)

            self.get_logger().info(f"Received Bounding Box: {self.bb_item}")
            center_x = (self.bb_x1 + self.bb_x2) / 2
            image_center_x = 600
            diff_x = image_center_x - center_x
            
            self.get_logger().info(f"Diff x: {diff_x}")

            if (abs(diff_x) > center_tolerance):
                self.current_twist.angular.z = float(min(max(diff_x/3000,-0.1),-0.05))
            else:
                self.get_logger().info(f"Facing Object")
                self.object_heading = math.radians(self.heading)

                #Stop turning.
                self.current_twist.angular.z = float(0.0)

                aligned = True

            self.twist_pub.publish(self.current_twist)

        self.get_logger().info(f"Distance to object: {self.min_forward}")

        #Just running wheels (ez mode)
        while float(self.min_forward) > 3:
            #Spin rclpy to update distance.
            rclpy.spin_once(self, timeout_sec=0.05)


            self.get_logger().info(f"Distance to object: {self.min_forward}")

            #Move forward
            self.current_twist.linear.x = float(0.3)
            self.twist_pub.publish(self.current_twist)
        
        #Stop the robot
        self.current_twist.linear.x = float(0.0)
        self.twist_pub.publish(self.current_twist) 

        self.get_logger().info(f"Reached object, taking photo.")   

        sleep(1)

        # Take photo
        self.take_photo()

        #Record object location
        self.get_logger().info(f"Appending current location to list and publishing marker")   

        #Get the current map-base_frame transform for x and y robot coordinates
        try:
            now = rclpy.time.Time()
            transform = self.tf_buffer.lookup_transform(
                "map",
                "base_link",  # Source frame
                now,
                timeout=rclpy.duration.Duration(seconds=5.0)
            )

            # Extract translation
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            z = transform.transform.translation.z  # optional

            # Extract yaw from quaternion
            q = transform.transform.rotation
            quaternion = [q.x, q.y, q.z, q.w]
            (_, _, yaw) = euler_from_quaternion(quaternion)

            # Use x, y (and yaw if needed)
            self.publish_goal_marker(x, y, int(time.time()), "r", delete=False)
            self.object_list.append((self.bb_item, x, y, self.heading))

            self.get_logger().info("Recorded map-frame position using TF")

        except TransformException as e:
            self.get_logger().error(f"Transform error: {str(e)}")

        # Resume normal path
        self.bounding_box_detected = False

    def explore(self):
        while not self.completed_exploration:

            sleep(1)
            #Spin a few times to get correct data
            while self.activated == False:
                rclpy.spin_once(self, timeout_sec=0.1)

            # if not self.activated:
            #     self.get_logger().info("Exploration not activated, waiting...")
            #     rclpy.spin_once(self, timeout_sec=0.1)
            #     continue

            if not self.goal_list:
                # First-time initialization
                x_min, x_max = -6, 6
                y_min, y_max = -6, 6
                step = 3
                self.goal_list = (
                    self.lawnmower1(x_min, x_max, 0, y_max, step)
                    + self.lawnmower2(x_min, x_max, y_min, y_max, step)
                    + self.lawnmower3(x_min, x_max, y_min, 0, step)
                )
                self.get_logger().info(f"Generated {len(self.goal_list)} goals")

            while self.explore_index < len(self.goal_list) and self.activated:
                x, y, theta = self.goal_list[self.explore_index]
                self.get_logger().info(f"Navigating to goal #{self.explore_index + 1}")
                self.publish_goal_marker(x, y, self.explore_index, "g", delete=False)

                future = self.send_goal(x, y, theta)
                rclpy.spin_until_future_complete(self, future)
                goal_handle = future.result()

                if not goal_handle.accepted:
                    self.get_logger().warn(f"Goal #{self.explore_index + 1} rejected")
                    self.explore_index += 1
                    continue

                self.get_logger().info(f"Goal #{self.explore_index + 1} accepted")
                result_future = goal_handle.get_result_async()
                start_time = time.time()

                while rclpy.ok() and not result_future.done():
                    rclpy.spin_once(self, timeout_sec=0.1)

                    if not self.activated:
                        self.get_logger().info("Exploration deactivated. Saving current index.")
                        goal_handle.cancel_goal_async()
                        break
                        #return

                    if self.bounding_box_detected:
                        self.get_logger().info("Goal paused due to bounding box detection.")
                        self.publish_goal_marker(x, y, self.explore_index, "g", delete=True)
                        goal_handle.cancel_goal_async()
                        self.clear_path()
                        self.handle_bounding_box()

                        self.get_logger().info("Resending the same goal...")
                        future = self.send_goal(x, y, theta)
                        rclpy.spin_until_future_complete(self, future)
                        goal_handle = future.result()
                        self.publish_goal_marker(x, y, self.explore_index, "g", delete=False)

                        if not goal_handle.accepted:
                            self.get_logger().warn("Resent goal was rejected")
                            break

                        result_future = goal_handle.get_result_async()
                        start_time = time.time()
                        continue

                    if time.time() - start_time > 45.0:
                        self.get_logger().warn(f"Goal #{self.explore_index + 1} timed out.")
                        goal_handle.cancel_goal_async()
                        self.clear_path()
                        break

                if result_future.done():
                    result = result_future.result().result
                    if result.error_code == 0:
                        self.get_logger().info(f"Goal #{self.explore_index + 1} succeeded")
                        self.clear_path()
                        self.publish_goal_marker(x, y, self.explore_index, "g", delete=True)
                    else:
                        self.get_logger().warn(
                            f"Goal #{self.explore_index + 1} failed - Code: {result.error_code}"
                        )
                        self.clear_path()
                        self.publish_goal_marker(x, y, self.explore_index, "g", delete=True)
                else:
                    self.get_logger().info(f"Goal #{self.explore_index + 1} not completed.")
                    self.clear_path()
                    self.publish_goal_marker(x, y, self.explore_index, "g", delete=True)

                self.explore_index += 1  # Only advance if we attempted the goal

            #Check to see if we have completed exploration
            if self.explore_index == (len(self.goal_list) - 1):
                self.completed_exploration = True
                self.get_logger().info("Exploration complete, driving to detected numbers...")

        self.drive_to_points(self)


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
        route_index = 1
        completed_exploration_waypoint_rerun = False

        while not completed_exploration_waypoint_rerun:
            sleep(1)

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
            

    def make_pose(self, x, y):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.w = 1.0
        return pose

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
    #node.explore()
    node.drive_to_points()
    node.destroy_node()
    rclpy.shutdown()

# def main(args=None):
#     rclpy.init(args=args)
#     node = AutoNavigator()

#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()


if __name__ == '__main__':
    main()