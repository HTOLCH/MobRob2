import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import tkinter as tk
import os
from sensor_msgs.msg import Image
from PIL import Image as PILImage, ImageTk
from cv_bridge import CvBridge
import cv2 
import threading
from sensor_msgs.msg import NavSatFix
import math
from std_msgs.msg import String
import serial
from std_msgs.msg import Float32MultiArray

os.environ["DISPLAY"] = ":1"
#Run: 
# ros2 run joy_linux joy_linux_node
#To publish joystick information


#Master class
class Master(Node):

    #initialize paremeters
    def __init__(self):
        super().__init__('master')

        self.bridge = CvBridge()

        self.text_size = 14

        #set up display
        self.root = tk.Tk()
        self.root.title("Mode Display")

        self.label = tk.Label(self.root, text=f"Mode: Deactivated", font=("Helvetica", self.text_size))
        self.label.pack(pady=3, expand=True)

        self.heading_label = tk.Label(self.root, text=f"Heading: 0.0", font=("Helvetica", self.text_size))
        self.heading_label.pack(pady=3, expand=True)

        self.smallest_distance_label = tk.Label(self.root, text=f"Smallest Distance: 0.0", font=("Helvetica", self.text_size))
        self.smallest_distance_label.pack(pady=3, expand=True)

        self.lat_long_label = tk.Label(self.root, text=f"Lat: 0.0, Long: 0.0", font=("Helvetica", self.text_size))
        self.lat_long_label.pack(pady=3, expand=True)

        self.avoidance_status_label = tk.Label(self.root, text=f"Avoidance Status: No objects detected", font=("Helvetica", self.text_size))
        self.avoidance_status_label.pack(pady=3, expand=True)

        self.cbDistances_label = tk.Label(self.root, text=f"Cone-Bucket Distances: []", font=("Helvetica", self.text_size))
        self.cbDistances_label.pack(pady=3, expand=True)

        self.operation_state_label = tk.Label(self.root, text=f"Operation State: SEND HELP", font=("Helvetica", self.text_size))
        self.operation_state_label.pack(pady=3, expand=True)

        # Add a canvas to display the camera image
        self.canvas_width = 640
        self.canvas_height = 480
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack(pady=10)

        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.auto_status_publisher = self.create_publisher(Bool, 'auto_status', 10)
        self.skip_map_publisher = self.create_publisher(Bool, 'skip_mapping', 10)

        # Subscriber to the 'joy' topic
        self.joy_subscription = self.create_subscription(
            Joy,
            'joy',
            self.joy_callback,
            10
        )

        #Subsciber to auto joy topic for automated movement
        self.auto_cmd_subscription = self.create_subscription(
            Twist,
            'auto_joy',
            self.auto_joy_callback,
            10
        )

        #Subscriber to camera topic
        #self.camera_subscription = self.create_subscription(
        #    Image,
        #    '/oak/rgb/image_raw',
        #    self.image_callback,
        #    10
        #)

        #Subscriber to the image with bboxes topic
        #Subscriber to camera topic
        self.image_bb = self.create_subscription(
            Image,
            '/image_with_bboxes',
            self.image_callback,
            10
        )


        self.phidget_subscription = self.create_subscription(
            Float64,
            'phidget',
            self.phidget_callback,
            10
        )

        #Subscriber to min distance topic
        self.min_distance_subscription = self.create_subscription(
            String,
            'distances',
            self.min_distance_callback,
            10
        )

        # Subscriber to the 'fix' topic
        self.fix_subscription = self.create_subscription(
            NavSatFix,
            '/fix',
            self.gps_callback,
            10
        )

        # Subscriber to the 'avoidance_status' topic
        self.avoidance_status_subscription = self.create_subscription(
            String,
            '/avoidance_status',
            self.avoidance_status_callback,
            1
        )

        self.cbDistances_subscription = self.create_subscription(
            Float32MultiArray,
            '/cbDistances',
            self.cbDistances_callback,
            1
        )

        self.operation_state_subscription = self.create_subscription(
            String,
            '/operation_state',
            self.operation_state_callback,
            1
        )

        self.timer = self.create_timer(0.1, self.timer_callback)

        self.previous_buttons = []  # Track previous button states
        self.get_logger().info('Master has been started.')

        self.current_twist = Twist()
        
        self.current_heading = 0.0

        self.smallest_distance = 0.0

        self.manual_mode = False
        self.automatic_mode = False
        self.super_saiyan = False
        self.inside = False

        self.current_image = None
        self.image_on_canvas = None
        self.image_updated = False  # Flag to indicate new image is ready
        self.pil_image = None

        # Initialize current position and heading
        self.current_lat = 0.0
        self.current_lon = 0.0

        self.avoidance_status = "No objects detected"

        self.cbDistances = []

        self.operation_state = "T"

        self.skip_mapping = False

        self.skip_publish_counter = 0

        # Start periodic updates
        self.update_gui()


    def joy_callback(self, msg):

        #dead man switch condition for manual or automated mode
        if msg.buttons[7] == 1:
            self.inside = True

            if msg.buttons[3] == 1:
                self.get_logger().info("Skipping mapping...")
                self.skip_mapping = True
            else:
                self.skip_mapping = False
                
            #self.get_logger().info(f"X Axes: {-msg.axes[0]}")
            #self.get_logger().info(f"Y Axes: {msg.axes[1]}")

            # Initialize previous_buttons to match the number of buttons on the joystick
            if not self.previous_buttons:
                self.previous_buttons = [0] * len(msg.buttons)
            
            for i, button_state in enumerate(msg.buttons):
                # Check for a button press-and-release event
                if self.previous_buttons[i] == 1 and button_state == 0:
                    if i == 0:
                        self.get_logger().info("Entering Automated Mode")
                        self.manual_mode = False
                        self.automatic_mode = True
                        self.super_saiyan = False
                    if i == 1:
                        self.get_logger().info("Entering Manual Mode")
                        self.manual_mode = True
                        self.automatic_mode = False  
                    if self.manual_mode:
                        if i == 2:
                            if not self.super_saiyan:
                                self.get_logger().info("Entering Super Saiyan Mode")
                                self.super_saiyan = True
                            else:
                                self.get_logger().info("Exiting Super Saiyan Mode")
                                self.super_saiyan = False


            if self.manual_mode:
                if self.super_saiyan:
                    forward = (msg.axes[1]*2) + (msg.axes[7]/4) #Allow for fine forward turning with left buttons
                    turn = msg.axes[0] + (msg.axes[6]/4)  #Allow for fine turn turning with left buttons
                    self.current_twist.linear.x = forward  
                    self.current_twist.angular.z = turn

                else:
                    forward = (msg.axes[1]/2) + (msg.axes[7]/4) #Allow for fine forward turning with left buttons
                    turn = msg.axes[0] + (msg.axes[6]/4)  #Allow for fine turn turning with left buttons
                    self.current_twist.linear.x = forward  
                    self.current_twist.angular.z = turn
            
        #only runs once after dead man switch is released
        elif self.inside:
            self.get_logger().info("Dead man switch triggered")
            self.inside = False
            self.manual_mode = False
            self.automatic_mode = False
            self.current_twist.linear.x = float(0)
            self.current_twist.angular.z = float(0)
        
        
        # Update the previous_buttons to the current state
        self.previous_buttons = msg.buttons

    def auto_joy_callback(self, msg):
        #self.get_logger().info(f"forward: {msg.linear.x}\nturn: {msg.angular.z}")
        if self.automatic_mode:
            self.current_twist.linear.x = msg.linear.x
            self.current_twist.angular.z = msg.angular.z

    def phidget_callback(self, msg):
        #self.get_logger().info(f"Phidget: {msg.data}")
        self.current_heading = msg.data

    def gps_callback(self, msg):
        # Check if the latitude and longitude are valid
        if not math.isnan(msg.latitude) and not math.isnan(msg.longitude):
            self.current_lat = msg.latitude
            self.current_lon = msg.longitude
            #self.get_logger().info(f"Received valid GPS: ({self.current_lat}, {self.current_lon})")
            #self.drive_to_waypoint()  # Only proceed if valid GPS data is available
        else:
            #self.get_logger().warn("Invalid GPS data received. Waiting for valid data...")
            self.current_lat = 0.0
            self.current_lon = 0.0

    def min_distance_callback(self, msg):
        self.smallest_distance = msg.data

    def avoidance_status_callback(self,msg):
        self.avoidance_status = msg.data

    def cbDistances_callback(self,msg):
        self.cbDistances = list(msg.data)
        #self.get_logger().info(f"cbDistances: {self.cbDistances}")

    def operation_state_callback(self,msg):
        self.operation_state = msg.data
        if self.operation_state == "T":
            self.operation_state = "In transit"
        elif self.operation_state == "C":
            self.operation_state = "Searching for the waypoint"
        elif self.operation_state == "CC":
            self.operation_state = "Moving to within the cone region"
        elif self.operation_state == "B":
            self.operation_state = "Searching for the object"
        elif self.operation_state == "S":
            self.operation_state = "Saving recorded data"
        elif self.operation_state == "L":
            self.operation_state = "Performing exit maneuver"
        elif self.operation_state == "F":
            self.operation_state = "All tasks finished"

        #self.get_logger().info(f"Operation State: {self.operation_state}")

    def image_callback(self, msg):
        # Convert ROS image to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Resize the image to fit the canvas while maintaining aspect ratio
        cv_image_resized = self.resize_image(cv_image)

        cv_image_rgb = cv2.cvtColor(cv_image_resized, cv2.COLOR_BGR2RGB)

        # Convert the image to a PIL Image to use with Tkinter
        self.pil_image = PILImage.fromarray(cv_image_rgb)
        #self.current_image = ImageTk.PhotoImage(pil_image)

        # Update the image flag to indicate new image is ready
        self.image_updated = True
        #self.update_canvas_image()

    def resize_image(self, cv_image):
        # Get the original image dimensions
        height, width = cv_image.shape[:2]

        # Calculate the aspect ratio
        aspect_ratio = width / height

        # Calculate the new dimensions
        new_width = self.canvas_width
        new_height = int(new_width / aspect_ratio)

        #ADDED TO CHANGE ASPECT
        #new_height = self.canvas_height
        #new_width = int(new_height * aspect_ratio)

        if new_height > self.canvas_height:
            new_height = self.canvas_height
            new_width = int(new_height * aspect_ratio)

        # Resize the image while maintaining the aspect ratio
        return cv2.resize(cv_image, (new_width, new_height))

    def timer_callback(self):
        #Telling the autoDrive node whether it is allowed to run or not
        msg = Bool()
        msg.data = self.automatic_mode
        self.cmd_vel_publisher.publish(self.current_twist)
        self.auto_status_publisher.publish(msg)

        if self.skip_mapping:
            if self.skip_publish_counter < 10:   
                self.skip_publish_counter +=1
                msg = Bool()
                msg.data = True
                self.skip_map_publisher.publish(msg)
            else:
                #Stop publishing.
                self.skip_publish_counter = 0
                self.skip_mapping = False
        else:
            msg = Bool()
            msg.data = False
            self.skip_map_publisher.publish(msg)

    def update_canvas_image(self):
        # If the image already exists on the canvas, update it, otherwise create a new one
        if self.image_on_canvas:
            self.canvas.itemconfig(self.image_on_canvas, image=self.current_image)
        else:
            self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_image)

    def update_gui(self):
        if self.image_updated and hasattr(self, 'pil_image'):
            self.current_image = ImageTk.PhotoImage(self.pil_image)
            self.update_canvas_image()
            self.image_updated = False  # Reset the flag after updating the canvas

        # Update GUI label based on mode
        if self.manual_mode:
            if self.super_saiyan:
                self.label.config(text=f"Mode: Manual\nSuper Saiyan", fg="yellow")
            else:
                self.label.config(text=f"Mode: Manual", fg="orange")
        elif self.automatic_mode:
            self.label.config(text=f"Mode: Automatic", fg="blue")
        else:
            self.label.config(text=f"Mode: Deactivated", fg="red")

        self.heading_label.config(text=f"Heading: {self.current_heading:.2f}", fg = "black")

        self.smallest_distance_label.config(text= self.smallest_distance, fg = "black")

        self.lat_long_label.config(text=f"Lat: {self.current_lat:.2f}, Long: {self.current_lon:.2f}", fg = "black")

        self.avoidance_status_label.config(text=f"{self.avoidance_status}", fg = "black")

        self.cbDistances_label.config(text=f"Cone-Bucket Distances: {self.cbDistances}", fg = "black")

        self.operation_state_label.config(text=f"Operation State: {self.operation_state}", fg = "black")

        # Schedule the next GUI update
        self.root.after(10, self.update_gui)

    def run_tkinter(self):
        self.root.mainloop()


def ros_thread(node):
    rclpy.spin(node)



def main(args=None):
    rclpy.init(args=args)
    node = Master()

    ros_thread_obj = threading.Thread(target=ros_thread, args=(node,))
    ros_thread_obj.start()

    try:
        #rclpy.spin(node)
        node.run_tkinter()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        ros_thread_obj.join()

if __name__ == '__main__':
    main()
