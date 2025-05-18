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


class AutoNavigator(Node):
    def __init__(self):
        super().__init__('auto_navigator')
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Create a client for the 'take_photo' service
        self.camera_client = self.create_client(TakePhoto, 'take_photo')

        self.distances_publisher = self.create_publisher(String, 'distances', 10)
        self.twist_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_pub = self.create_publisher(Marker, '/visualization_marker', 10)

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

        self.completed_exploration = False

        #Bounding box subscription
        self.subscription = self.create_subscription(
            String,  # Replace with the actual bounding box message type
            'bounding_box',
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

        self.get_logger().info("auto node up")


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

    def publish_goal_marker(self, x, y, idx, delete=False):
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
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        self.marker_pub.publish(marker)


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
        self.get_logger().info("Bounding box detected. Interrupting current path.")
        
        #Bounding boxes will be published as a list: [x1,x2,y1,y2,item]
        data = list(msg.data)

        self.bb_x1 = data[0]
        self.bb_x2 = data[1]
        self.bb_y1 = data[2]
        self.bb_y2 = data[3]
        self.bb_item = data[4]

        #Only set true if the number/ object detected has not already been stored:
        if self.bb_item in self.object_list:
            self.get_logger().info("Detected object has already been stored, ignoring.")         
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
        while self.min_forward > 3:
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
            self.publish_goal_marker(x, y, int(time.time()), delete=False)
            self.object_list.append((self.bb_item, x, y))

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
                self.publish_goal_marker(x, y, self.explore_index, delete=False)

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
                        self.publish_goal_marker(x, y, self.explore_index, delete=True)
                        goal_handle.cancel_goal_async()
                        self.handle_bounding_box()

                        self.get_logger().info("Resending the same goal...")
                        future = self.send_goal(x, y, theta)
                        rclpy.spin_until_future_complete(self, future)
                        goal_handle = future.result()
                        self.publish_goal_marker(x, y, self.explore_index, delete=False)

                        if not goal_handle.accepted:
                            self.get_logger().warn("Resent goal was rejected")
                            break

                        result_future = goal_handle.get_result_async()
                        start_time = time.time()
                        continue

                    if time.time() - start_time > 45.0:
                        self.get_logger().warn(f"Goal #{self.explore_index + 1} timed out.")
                        goal_handle.cancel_goal_async()
                        break

                if result_future.done():
                    result = result_future.result().result
                    if result.error_code == 0:
                        self.get_logger().info(f"Goal #{self.explore_index + 1} succeeded")
                        self.publish_goal_marker(x, y, self.explore_index, delete=True)
                    else:
                        self.get_logger().warn(
                            f"Goal #{self.explore_index + 1} failed - Code: {result.error_code}"
                        )
                        self.publish_goal_marker(x, y, self.explore_index, delete=True)
                else:
                    self.get_logger().info(f"Goal #{self.explore_index + 1} not completed.")
                    self.publish_goal_marker(x, y, self.explore_index, delete=True)

                self.explore_index += 1  # Only advance if we attempted the goal



    def explore_old(self):
        if not self.activated:
            self.get_logger().info("Exploration not activated.")
            return

        # Define a list of goals to explore
        #Grid bounds
        x_min, x_max = -6, 6
        y_min, y_max = -6, 6
        step = 1  # Go every 3 units
        
        goal_list = self.lawnmower1(x_min,x_max,0,y_max,step)

        goal_list2 = self.lawnmower2(x_min,x_max,y_min,y_max,step)

        goal_list3 = self.lawnmower3(x_min,x_max,y_min,0,step)

        goal_list = goal_list + goal_list2 + goal_list3

        self.get_logger().info(f"Goals: {goal_list}")

        for idx, (x, y, theta) in enumerate(goal_list):
            self.get_logger().info(f"Navigating to goal #{idx + 1}")
            self.publish_goal_marker(x, y, idx, delete=False)

            future = self.send_goal(x, y, theta)
            #rclpy.spin_until_future_complete(self, future)
            goal_handle = future.result()

            if not goal_handle.accepted:
                self.get_logger().warn(f"Goal #{idx + 1} rejected")
                continue

            self.get_logger().info(f"Goal #{idx + 1} accepted")

            result_future = goal_handle.get_result_async()

            # Wait up to 15 seconds for result
            start_time = time.time()
            while rclpy.ok() and not result_future.done():
                rclpy.spin_once(self, timeout_sec=0.1)

                if not self.activated:
                    self.get_logger().info("Exploration deactivated during goal. Cancelling goal.")
                    goal_handle.cancel_goal_async()
                    return

                if self.bounding_box_detected:
                    self.get_logger().info("Goal paused due to bounding box detection.")
                    goal_handle.cancel_goal_async()
                    self.handle_bounding_box()

                    # Re-issue the same goal after handling
                    self.get_logger().info("Resending the same goal after bounding box handled...")

                    future = self.send_goal(x, y, theta)
                    rclpy.spin_until_future_complete(self, future)
                    goal_handle = future.result()

                    if not goal_handle.accepted:
                        self.get_logger().warn("Resent goal was rejected")
                        break

                    result_future = goal_handle.get_result_async()
                    start_time = time.time()  # Reset timer after resending
                    continue  # Stay in the while loop


                if time.time() - start_time > 45.0:
                    self.get_logger().warn(f"Goal #{idx + 1} timed out after 30s. Skipping...")
                    goal_handle.cancel_goal_async()
                    break

            if result_future.done():
                result = result_future.result().result
                if result.error_code == 0:
                    self.get_logger().info(f"Goal #{idx + 1} succeeded")
                    self.publish_goal_marker(x, y, idx, delete=True)
                else:
                    self.get_logger().warn(f"Goal #{idx + 1} failed - Error code: {result.error_code}, Message: {result.error_msg}")
            else:
                self.get_logger().info(f'Goal #{idx + 1} was not completed in time.')
                self.publish_goal_marker(x, y, idx, delete=True)


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
    node.explore()
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
