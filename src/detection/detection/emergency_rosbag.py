import rclpy
from rclpy.node import Node
from collections import deque
import time
import os
import sys

from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions
from rclpy.serialization import serialize_message
from rosidl_runtime_py.utilities import get_message


# sudo apt install ros-jazzy-rosbag2-py


class RosbagPreEventRecorder(Node):
    def __init__(self):
        super().__init__('rosbag_pre_event_recorder')
        self.buffers = {}  
        self.subscriptions = []
        self.buffer_duration_sec = 5.0  # seconds

        self.trigger_topic = '/autodrive/emergency_signal' # FIX -> what data type? (Bool?)

        # Output directory
        self.output_directory = '/workspace/rosbag'
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
            self.get_logger().info(f"Created output directory: {self.output_directory}")

        # Topics to explicitly include, if you don't want to record *all* topics

        self.included_topics = ['/odom', '/tf', '/diagnostics'] #...

        # Subscribe to the trigger topic
        self.create_subscription(
            self.trigger_topic,
            1 # only need most recent
        )
    
        # Set up subscriptions for all other topics to maintain buffers
        topic_list = self.get_topic_names_and_types() # get all topics and types
        self.get_logger().info("Subscribing to topics for buffering...")
        for topic_name, types in topic_list:

            if topic_name in self.included_topics:

                msg_type_str = types[0] # Take the first message type string

                try:
                    msg_type_class = get_message(msg_type_str)
                except Exception as e:
                    self.get_logger().warn(f"Could not load message type '{msg_type_str}' for topic '{topic_name}': {e}. Skipping subscription.")
                    continue # Skip this topic if its message type can't be loaded

                # Create subscription
                sub = self.create_subscription(
                    msg_type_class,
                    topic_name,
                    lambda msg, t=topic_name, m_type_str=msg_type_str: self.callback(t, msg, m_type_str),
                    10 # QoS history depth
                )
                self.subscriptions.append(sub)
                self.buffers[topic_name] = deque() # Initialize deque
                self.get_logger().info(f"Buffering topic: {topic_name} ({msg_type_str})")


    def trigger_callback(self, msg):
        # Assuming msg is a std_msgs/Bool where True means emergency
        if msg.data:
            if not self.triggered_at_least_once:
                self.get_logger().fatal("EMERGENCY SIGNAL RECEIVED! Saving buffered data to rosbag!")
                self.save_buffers_to_rosbag()
                self.triggered_at_least_once = True # Set flag to prevent repeated saving for continuous triggers
        else:
            if self.triggered_at_least_once:
                self.get_logger().info("Emergency signal cleared.")
                self.triggered_at_least_once = False # Reset flag when signal goes low

    def callback(self, topic, msg, msg_type_str):
        now_ns = self.get_clock().now().nanoseconds
        buf = self.buffers[topic]
        buf.append((now_ns, msg, msg_type_str))

        # Remove old messages
        cutoff_ns = now_ns - self.buffer_duration_sec
        while buf and buf[0][0] < cutoff_ns:
            buf.popleft()

    def save_buffers_to_rosbag(self):
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        bag_dir_name = f"emergency_record_{timestamp_str}"
        full_bag_path = os.path.join(self.output_directory, bag_dir_name)

        writer = SequentialWriter()
        
        # Configure storage options
        storage_options = StorageOptions(
            uri=full_bag_path,
            storage_id='sqlite3' # Default storage plugin (???)
        )
        # Configure converter options (usually default is fine)
        converter_options = ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )

        try:
            writer.open(storage_options, converter_options)
            self.get_logger().info(f"Opened rosbag writer to: {full_bag_path}")

            # Iterate through all topics that have buffered messages
            for topic_name, messages in self.buffers.items():
                if not messages:
                    continue # Skip empty buffers

                # Get the message type string from the first message in the buffer
                # Stored in the callback (timestamp_nanoseconds, message_object, msg_type_string)
                msg_type_str = messages[0][2] 
                
                # Create the topic in the bag file metadata
                writer.create_topic(
                    (topic_name, msg_type_str, 'cdr', '') # Topic metadata: name, type, serialization format, topic_group
                )   
                self.get_logger().info(f"Writing {len(messages)} messages for topic '{topic_name}' to rosbag.")

                # Write each message from the buffer to the rosbag
                for timestamp_ns, msg_obj, _ in messages: # msg_type_str is already known
                    serialized_msg = serialize_message(msg_obj)
                    writer.write(
                        topic_name,
                        serialized_msg,
                        timestamp_ns
                    )
            
            # Close the writer to finalize the bag file
            writer.close()
            self.get_logger().info(f"Successfully saved all buffered data to rosbag: {full_bag_path}")

        except Exception as e:
            self.get_logger().error(f"Failed to save buffers to rosbag at '{full_bag_path}': {e}")
            # It's a good idea to try to close the writer even if an error occurs
            try:
                writer.close()
            except Exception as close_e:
                self.get_logger().error(f"Error while closing rosbag writer after a failure: {close_e}")


    def destroy_node(self):
        self.get_logger().info("Shutting down recorder node.")
        super().destroy_node()

def main():
    rclpy.init()
    node = RosbagPreEventRecorder()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Recorder node manually interrupted.")
        # If you want to save on manual interrupt too, uncomment this line:
        # node.save_buffers_to_rosbag()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()