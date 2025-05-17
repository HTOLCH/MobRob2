/*  
*   A basic node for ros2 that runs with ariaCoda
*   To run use 'ros2 run ariaNode ariaNode -rp /dev/ttyUSB0'
*
*   Author: Kieran Quirke-Brown
*   Date: 12/01/2024
*/
 
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <signal.h>
 
// #include <rclcpp/rclcpp.hpp>
#include <tf2/LinearMath/Quaternion.hpp>
 
// #include <geometry_msgs/msg/twist.hpp>
// #include <std_msgs/msg/bool.hpp>  // Added for status publishing
// #include <std_msgs/msg/string.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
 
 
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
 
# include "Aria/Aria.h"
 
//used with signal handler as signal handler function doesn't accept parameters
bool stopRunning = false;
 
using namespace std::chrono_literals;
/*
*   Basic ROS node that updates velocity of pioneer robot, Aria doesn't like
*   being spun as a node therefore we just use a single subscriber
*   parameters:
*       forward and ratation speeds are float that are bound to the node
*       but point at the same location as the aria velocities
*/
class ariaNode : public rclcpp::Node {
    public:
        ariaNode(float* forwardSpeed, float* rotationSpeed, ArRobot* robot) : Node("Aria_node") {
            currentForwardSpeed = forwardSpeed;
            currentRotationSpeed = rotationSpeed;
            robot_ = robot;  // Store the robot pointer
 
            cmdVelSub = create_subscription<geometry_msgs::msg::Twist> (
                "cmd_vel", 10, std::bind(&ariaNode::cmdVelCallback, this, std::placeholders::_1)
            );
            // Publisher for robot odometry
            odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("bot/odom", 10);
 
            // Timer to periodically publish odometry (20Hz)
            odom_timer_ = this->create_wall_timer(
                std::chrono::milliseconds(50),
                std::bind(&ariaNode::publish_odom, this)
            );
        }
 
    private:
        void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg) {
           
            double linearSpeed = msg->linear.x;
            double angularSpeed = msg->angular.z;
 
            *currentForwardSpeed = linearSpeed;
            *currentRotationSpeed = angularSpeed;
 
            RCLCPP_DEBUG(this->get_logger(), "message received.");
 
        }
        void publish_odom() {
 
            if (robot_ != nullptr && robot_->isRunning()) {
                nav_msgs::msg::Odometry odom_msg;
                auto pose = robot_->getPose();
                auto time = this->now();
 
                geometry_msgs::msg::Point point;
                point.x = pose.getX()/1000; // Needs to be in meters
                point.y = pose.getY()/1000;
                point.z = 0; // Assume 2D
                tf2::Quaternion tf2_quat;
                tf2_quat.setRPY(0, 0, pose.getThRad());
                auto msg_quat = tf2::toMsg(tf2_quat);
               
               
                odom_msg.header.frame_id = "odom";
                odom_msg.header.stamp = time;
                odom_msg.child_frame_id = "base_link";
                odom_msg.pose.pose.position = point;
                odom_msg.pose.pose.orientation = msg_quat;
 
                odom_pub_->publish(odom_msg);
            }
           
        }
 
        rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmdVelSub;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
        rclcpp::TimerBase::SharedPtr odom_timer_;
        float* currentForwardSpeed;
        float* currentRotationSpeed;
        ArRobot* robot_;  // Add a member variable for the robot
};
 
// Deals with ctl+c handling to stop the motors correctly.
void my_handler(int s){
           printf("Caught signal %d\n",s);
           stopRunning = true;
}
 
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
 
    Aria::init();
    ArArgumentParser parser(&argc, argv);
    parser.loadDefaultArguments();
    ArRobot* robot;
    robot = new ArRobot();
 
    signal(SIGINT, my_handler);
   
    // RCLCPP_DEBUG(this->get_logger(),"Trying to connect to robot...");
    ArRobotConnector robotConnector(&parser, robot);
    if(!robotConnector.connectRobot()) {
        ArLog::log(ArLog::Terse, "simpleConnect: Could not connect to the robot.");
        if(parser.checkHelpAndWarnUnparsed()) {
            Aria::logOptions();
            Aria::exit(1);
        }
    }
 
    robot->setAbsoluteMaxTransVel(1500);
 
    float forwardSpeed = 0.0;
    float rotationSpeed = 0.0;
   
   
    // RCLCPP_DEBUG(aNode->get_logger(),"Run Async");
    robot->runAsync(true);
    // RCLCPP_DEBUG(aNode->get_logger(),"Enable Motors");
    robot->enableMotors();
 
    auto aNode = std::make_shared<ariaNode>(&forwardSpeed, &rotationSpeed, robot);
    RCLCPP_DEBUG(aNode->get_logger(),"Before Spin!...");
 
    /*
     *   Aria does not like to run in a ros node therefore we run a while loop
     *   that continuously spins the node to update velocities which are
     *   then sent using the normal Aria commands.
    */
    while (!stopRunning) {
        rclcpp::spin_some(aNode);
        // RCLCPP_DEBUG(aNode->get_logger(), "sending motor command.");
            robot->lock();
            robot->setVel(forwardSpeed * 500);
            robot->setRotVel(rotationSpeed * 50);
            robot->unlock();
            // RCLCPP_DEBUG(aNode->get_logger(), "motor command sent.");
            // RCLCPP_DEBUG(aNode->get_logger(), "forward speed is now %f.", forwardSpeed);
            // RCLCPP_DEBUG(aNode->get_logger(), "rotational speed is now %f.", rotationSpeed);
    }
    RCLCPP_DEBUG(aNode->get_logger(), "After Spin");
 
    robot->disableMotors();
    robot->stopRunning();
    // wait for the thread to stop
    robot->waitForRunExit();
 
    // exit
    RCLCPP_DEBUG(aNode->get_logger(), "ending Aria node");
    Aria::exit(0);
    return 0;
}