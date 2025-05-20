import rclpy
from rclpy.node import Node
from Phidget22.Phidget import *
from Phidget22.Devices.Spatial import *
import time
from std_msgs.msg import Float64
import signal
from std_srvs.srv import Trigger
from sensor_msgs.msg import Imu
import math
from tf_transformations import euler_from_quaternion

# Define the ROS 2 Node
class SpatialPublisher(Node):
    def __init__(self):
        super().__init__('spatial_publisher')
        self.publisher_ = self.create_publisher(Float64, 'phidget', 10)


        # Subscriber to the 'phidget' topic
        self.imu_subscription = self.create_subscription(
            Imu,
            '/imu/data_raw',
            self.imu_callback,
            10
        )

        self.get_logger().info('Phidget node initialized')

    def imu_callback(self, msg):
        # Extract quaternion from the IMU message
        quaternion = msg.orientation
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        roll, pitch, yaw = self.quaternion_to_euler2(quaternion)

        # Convert yaw (heading) to degrees
        heading_in_degrees = math.degrees(yaw) + 235 - 37 - 53 - 65 + 20
        self.publish_spatial_data(heading_in_degrees)
        # Print the heading value (in degrees)
        #self.get_logger().info(f"IMU Heading: {heading_in_degrees:.2f} degrees")


    def quaternion_to_euler(self, quaternion):
        """
        Convert a quaternion to Euler angles (roll, pitch, yaw).
        """
        import math
        # Extract quaternion components
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w

        # Roll (rotation around x-axis)
        sinr_cosp = +2.0 * (w * x + y * z)
        cosr_cosp = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (rotation around y-axis)
        sinp = +2.0 * (w * y - z * x)
        pitch = math.asin(sinp) if abs(sinp) <= 1 else math.copysign(math.pi / 2, sinp)

        # Yaw (rotation around z-axis)
        siny_cosp = +2.0 * (w * z + x * y)
        cosy_cosp = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw
    
    def quaternion_to_euler2(self,quaternion):
        orientation_list = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)

        return roll, pitch, yaw


    def publish_spatial_data(self, heading):
        message = Float64()
        message.data = heading
        self.publisher_.publish(message)

    # Callback for spatial data
    ##def on_algorithm_data(self, quaternion, timestamp):
    #    eulerAngles = self.spatial_device.getEulerAngles()
    #    # Publish the data to the topic
    #    self.publish_spatial_data(eulerAngles)

def main():
    rclpy.init()

    spatial_publisher = SpatialPublisher()

    try:
        rclpy.spin(spatial_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        spatial_device.close()
        spatial_publisher.destroy_node()
        rclpy.shutdown()

main()