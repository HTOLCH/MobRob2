import rclpy
from rclpy.node import Node
import tf2_ros
import geometry_msgs.msg
import nav_msgs.msg
import std_msgs.msg
import math
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Quaternion, Twist
from std_msgs.msg import Header

class TransformBroadcaster(Node):
    def __init__(self):
        super().__init__('transform_broadcaster')
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.odom2_publisher = self.create_publisher(Odometry, 'odom2', 10)

        # Subscribe to odometry for linear odom (from simulated wheel ticks)
        # self.subscription = self.create_subscription(
        #     Twist,
        #     '/cmd_vel',  # Change this if your odometry topic is different
        #     self.cmd_vel_callback,
        #     10
        # )

        self.subscription = self.create_subscription(
            std_msgs.msg.Float64,
            '/phidget',  # Change this if your odometry topic is different
            self.phidget_callback,
            10
        )

        self.subscription = self.create_subscription(
            Odometry,
            '/bot/odom',  # Change this if your odometry topic is different
            self.bot_odom,
            10
        )

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.prev_x = None
        self.prev_y = None

        self.last_time = self.get_clock().now()
        
        self.linear_vel = 0.0
        self.angular_vel = 0.0

        self.timer = self.create_timer(0.1, self.transform_publisher)

    def bot_odom(self,msg):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        if self.prev_x is not None and dt > 0:
            dx = x - self.prev_x
            dy = y - self.prev_y

            # Direction vector of heading
            vx_odom = math.sin(math.radians(self.yaw))
            vy_odom = math.cos(math.radians(self.yaw))

            # Project dx/dy along heading
            dot = (dx * vx_odom + dy * vy_odom)
            self.linear_vel = dot / dt  # Only forward velocity

            #dist = math.sqrt(dx**2 + dy**2)
            #self.linear_vel = dist / dt

        self.prev_x = x
        self.prev_y = y

        #YAYY THIS WORKING HEHEEH (avert ur eyes Kieran)
        self.y = x
        self.x = -y

    def phidget_callback(self,msg):
        self.yaw = -msg.data
        #self.get_logger().info(f"IMU Heading: {self.yaw:.2f} degrees")

    # def cmd_vel_callback(self, msg):
    #     self.linear_vel = msg.linear.x
    #     self.angular_vel = msg.angular.z
    
    def transform_publisher(self):    
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now
        

        #self.get_logger().info(f"Current linearvel: {self.linear_vel}")
        #self.get_logger().info(f"Current Heading: {self.yaw}")

        # Compute velocities in odom frame
        vx_odom = self.linear_vel * math.cos(math.radians(self.yaw))
        vy_odom = self.linear_vel * math.sin(math.radians(self.yaw))

        # Integrate position
        #self.x += vx_odom * dt
        #self.y += vy_odom * dt

        # Convert yaw to quaternion
        q = quaternion_from_euler(0, 0, math.radians(self.yaw))

        # Publish Odometry message
        odom_msg = Odometry()
        odom_msg.header = Header()
        odom_msg.header.stamp = now.to_msg()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"

        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y = self.y
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        odom_msg.twist.twist.linear.x = self.linear_vel
        #odom_msg.twist.twist.angular.z = self.angular_vel

        self.odom2_publisher.publish(odom_msg)

        # Broadcast the transform
        t = TransformStamped()
        t.header = odom_msg.header
        t.child_frame_id = odom_msg.child_frame_id
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        t.transform.rotation = odom_msg.pose.pose.orientation

        #self.get_logger().info(f"Current Heading: {self.yaw:.2f}")

        self.tf_broadcaster.sendTransform(t)

    def odom_callback(self, msg):
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"

        # Get position from odometry message
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z

        # Get orientation
        t.transform.rotation.x = msg.pose.pose.orientation.x
        t.transform.rotation.y = msg.pose.pose.orientation.y
        t.transform.rotation.z = msg.pose.pose.orientation.z
        t.transform.rotation.w = msg.pose.pose.orientation.w

        self.tf_broadcaster.sendTransform(t)

def main():
    rclpy.init()
    node = TransformBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
