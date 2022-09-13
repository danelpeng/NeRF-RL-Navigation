"""
This is a simple gym-style gazebo env for turtlebot robot.
Author: 
"""
import rospy 
import tf 
import roslaunch
import rospkg 
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from gazebo_msgs.srv import GetModelState, SetModelState
from gazebo_msgs.msg import ModelState
from sensor_msgs.msg import Image
from gazebo_msgs.msg import ContactsState

import gym
from gym.spaces import Box
import envs.config as config
import time 
import numpy as np 
import math 
import os 
import cv2 
from cv_bridge import CvBridge

class GazeboSimulation(gym.Env):
    def __init__(self):
        super().__init__()
        self.maze_id = config.maze_id
        self.continuous = config.continuous
        self.goal_space = config.goal_space_maze1
        self.start_space = config.start_space_maze1
        #Launch the simulation with the given launch file name 
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        path = rospkg.RosPack().get_path('turtle_helper')
        self.launch = roslaunch.parent.ROSLaunchParent(uuid, [path + '/launch/kobuki_nav_gazebo.launch'])
        self.launch.start()

        rospy.init_node('env_node')
        time.sleep(10)

        self.vel_pub = rospy.Publisher('/kabuki/cmd_vel', Twist, queue_size=5)#'/kabuki/cmd_vel'?? /kobuki/cmd_vel
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics',Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics',Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.get_state = rospy.ServiceProxy('/gazebo/get_model_state',GetModelState)
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state',SetModelState)

        self.goal = []
        self.p = []
        self.reward = 0
        self.success = False

        self.img_height = config.input_dim[0]
        self.img_width = config.input_dim[1]
        self.img_channels = config.input_dim[2]
        self.action_space = Box(
            low=np.array([0.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        self.observation_space = Box(
            low=0,
            high=1,   #for collect image: 255; for train: 1
            shape=(48, 64, 3),
            dtype=np.float64,
        )
        self.vel_cmd = [0.0, 0.0]
    
    def close(self):
        self.launch.shutdown()
        time.sleep(10)
    
    def reset(self):
        """
        Reset environment and setup for new episode
        """
        start_idx = np.random.choice(len(self.start_space))
        goal_idx = np.random.choice(len(self.goal_space))
        start = self.start_space[start_idx]
        theta = np.random.uniform(0, 2.0*np.pi)
        self.set_start(start[0], start[1], theta)
        self.goal = self.goal_space[goal_idx]
        self.set_goal(self.goal[0], self.goal[1])
        d0, alpha0 = self.goal2robot(self.goal[0] - start[0], self.goal[1] - start[1], theta)
        self.p = [d0, alpha0]
        self.reward = 0

        self.success = False
        self.vel_cmd = [0. ,0. ]

        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_proxy
        except rospy.ServiceException:
            print("/gazebo/reset_simulation service call failed")


        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            rospy.ServiceProxy('/gazebo/unpause_physics',Empty)
        except rospy.ServiceException:
            print("/gazebo/unpause_physics service call failed")
        
        image_data = None
        cv_image = None
        while image_data is None:
            image_data = rospy.wait_for_message('/camera/rgb/image_raw',Image)
            cv_image = CvBridge().imgmsg_to_cv2(image_data,"bgr8")
        
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause
        except rospy.ServiceException:
            print("/gazebo/pause_physics service call failed")
        
        if self.img_channels == 1:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        cv_image = cv2.resize(cv_image, (self.img_width, self.img_height))

        cv_image = np.array(cv_image)

        obs = cv_image/255.0
        return obs 
    
    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause
        except rospy.ServiceException:
            print("/gazebo/unpause_physics service call failed")
        
        vel_cmd = Twist()
        if self.continuous:
            vel_cmd.linear.x = config.v_max*action[0]
            vel_cmd.angular.z = config.w_max*action[1]
        else:
            # 3 actions
            if action == 0:  # Left
                vel_cmd.linear.x = 0.25
                vel_cmd.angular.z = 1.0
            elif action == 1:  # H-LEFT
                vel_cmd.linear.x = 0.25
                vel_cmd.angular.z = 0.4
            elif action == 2:  # Straight
                vel_cmd.linear.x = 0.25
                vel_cmd.angular.z = 0
            elif action == 3:  # H-Right
                vel_cmd.linear.x = 0.25
                vel_cmd.angular.z = -0.4
            elif action == 4:  # Right
                vel_cmd.linear.x = 0.25
                vel_cmd.angular.z = -1.0
            else:
                raise Exception('Error discrete action: {}'.format(action))
        
        self.vel_cmd = []
        self.vel_pub.publish(vel_cmd)
        time.sleep(0.05)

        done = False
        self.reward = 0
        image_data = None
        cv_image = None

        while image_data is None:
            image_data = rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=5)
            cv_image = CvBridge().imgmsg_to_cv2(image_data,"bgr8")

        if self.img_channels == 1:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, (self.img_width, self.img_height))   

        cv_image = np.array(cv_image) 
        obs = cv_image/255.0
        
        contact_data = None
        while contact_data is None:
            contact_data = rospy.wait_for_message('/contact_state', ContactsState, timeout=5) 
        collision = contact_data.states != []
        if collision:
            done = True
            self.reward = config.r_collision

        robot_state = None
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            get_state = self.get_state
            robot_state = get_state("robot", "world")
            assert robot_state.success is True
        except rospy.ServiceException:
            print("/gazebo/get_model_state service call failed")
        
        position = robot_state.pose.position
        orientation = robot_state.pose.orientation
        d_x = self.goal[0] - position.x
        d_y = self.goal[1] - position.y

        _, _, theta = tf.transformations.euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w])
        d, alpha = self.goal2robot(d_x, d_y, theta)
        if d<config.Cd:
            done = True
            self.reward = config.r_arrive
            self.success = True
            print("arrival!")
        
        if not done:
            delta_d = self.p[0] - d
            self.reward = config.r_move*delta_d + config.r_time_penalty

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause
        except rospy.ServiceException:
            print("/gazebo/pause_physics service call failed")

        self.p = [d, alpha]

        info = dict(
            collided = collision,
            #time = self.current_time - self.start_time,
            success = self.success,
            relative_position = self.p,
            # camera_pos = np.array([camera_x, camera_y, camera_z]),
            # camera_quat = np.array([camera_orientation_x, camera_orientation_y, camera_orientation_z, camera_orientation_w]),
        )
        return obs, self.reward, done, info
    
    def goal2robot(self, d_x, d_y, theta):
        d = np.sqrt(d_x * d_x + d_y * d_y)
        alpha = math.atan2(d_y, d_x) - theta
        return d, alpha

    def set_start(self, x, y, theta):
        state = ModelState()
        state.model_name = 'robot'
        state.reference_frame = 'world'  # ''ground_plane'
        # pose
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = 0
        quaternion = tf.transformations.quaternion_from_euler(0, 0, theta)
        state.pose.orientation.x = quaternion[0]
        state.pose.orientation.y = quaternion[1]
        state.pose.orientation.z = quaternion[2]
        state.pose.orientation.w = quaternion[3]
        # twist
        state.twist.linear.x = 0
        state.twist.linear.y = 0
        state.twist.linear.z = 0
        state.twist.angular.x = 0
        state.twist.angular.y = 0
        state.twist.angular.z = 0

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = self.set_state
            result = set_state(state)
            assert result.success is True
        except rospy.ServiceException:
            print("/gazebo/get_model_state service call failed")

    def set_goal(self, x, y):
        state = ModelState()
        state.model_name = 'goal'
        state.reference_frame = 'world'  # ''ground_plane'
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = 0.1

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = self.set_state
            result = set_state(state)
            assert result.success is True
        except rospy.ServiceException:
            print("/gazebo/get_model_state service call failed")
