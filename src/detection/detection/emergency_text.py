import rclpy
from rclpy.node import Node
from collections import deque # double ended queue
import time
from rosidl_runtime_py.utilities import get_message

class TopicRecorder(Node):
    def __init__(self):
        super().__init__('topic_recorder')
        self.buffers = {}  # topic -> deque of (timestamp, message)
        self.subscriptions = []
        self.buffer_duration = 5.0  # seconds

        # Define a list of topics to exclude
        # probably try reduce because there will be a lot... 
        self.excluded_topics = [
        ]

        # Get all topic types
        topic_list = self.get_topic_names_and_types() # ros2 method.. returns each topic name and its message type

        for topic_name, types in topic_list:

            if topic_name in self.excluded_topics:
                self.get_logger().info(f"Excluding topic: {topic_name}")
                continue 

            msg_type = get_message(types[0]) # convert to string

            # subscribe to each topic.. 
            sub = self.create_subscription(
                msg_type,
                topic_name,
                lambda msg, t=topic_name: self.callback(t, msg), # ensure callback is triggered immediately 
                10
            )
            self.subscriptions.append(sub)
            self.buffers[topic_name] = deque()

    # called whenever message is received on any of the subscribed topics
    def callback(self, topic, msg): 
        now = time.time()
        buf = self.buffers[topic] # get the deque associated with topic
        buf.append((now, msg)) # append current timestamp and received message

        # Remove old messages from the left of the queue
        while buf and now - buf[0][0] > self.buffer_duration:
            buf.popleft()

    # called when node is gracefully shutdown
    # iterate through each topics message buffer and writes it to a textfile 
    def save_buffers(self):
        for topic, messages in self.buffers.items():
            with open(f'{topic.replace("/", "_")}.txt', 'w') as f:
                for ts, msg in messages:
                    f.write(f'{ts}: {msg}\n')

def main():
    rclpy.init()
    node = TopicRecorder()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Saving last 5 seconds of data...")
        node.save_buffers()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
