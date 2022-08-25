import gym
import time
import numpy as np
import os 
from os.path import join
import subprocess
from gym.spaces import Box
import sys

import cv2

try:
    import rospy
    import rospkg
    from geometry_msgs.msg import Twist
except ModuleNotFoundError:
    pass

sys.path.append(join(os.path.dirname(__file__),'..'))
from envs.gazebo_simulation import GazeboSimulation

class TurtleGazebo(gym.Env):
    def __init__(
        self,
        gui=True,
        init_sim=True,
        init_position=[0.0, -3.0, np.pi/2] ,    #[ 0.0, -3.0, np.pi/2]  
        goal_position=[2.5, 3.0, np.pi/2],
        max_step=10,
        time_step=1.0,    #1s == 1 time step
        success_reward=100,
        collision_reward=-50,
        move_reward=1,
    ):
        """
        Base RL env that initialize simulation in Gazebo
        """
        super().__init__()
        #config 
        self.gui = gui
        self.init_sim = init_sim
        self.init_position = init_position
        self.goal_position = goal_position
        self.time_step = time_step
        self.max_step = max_step
        self.step_count = 0
        self.collided = 0
        self.start_time = self.current_time = None

        #reward function
        self.success_reward = success_reward
        self.collision_reward = collision_reward
        self.move_reward = move_reward,

        #action
        min_v, max_v= 0.0, 1.0
        min_w, max_w=-1.0, 1.0
        self._cmd_vel_pub = rospy.Publisher('/kabuki/cmd_vel', Twist, queue_size=1)
        self.range_dict = RANGE_DCIT = {
            "linear_velocity": [min_v, max_v],
            "angular_velocity":[min_w, max_w],
        }
        self.action_space = Box(
            low=np.array([RANGE_DCIT["linear_velocity"][0], RANGE_DCIT["angular_velocity"][0]]),
            high=np.array([RANGE_DCIT["linear_velocity"][1], RANGE_DCIT["angular_velocity"][1]]),
            dtype=np.float32
        )

        #observation
        self.observation_space = Box(
            low=0,
            high=1,   #for collect image: 255; for train: 1
            shape=(96, 128, 3),
            dtype=np.float64,
        )

        #launch gazebo
        if init_sim:
            rospy.logwarn(">>>>>>>>>>>>>>>> Load world <<<<<<<<<<<<<")
            rospack = rospkg.RosPack()
            self.BASE_PATH = rospack.get_path('turtle_helper')
            launch_file = join(self.BASE_PATH,'launch','kobuki_nav_gazebo.launch')
            self.gazebo_process = subprocess.Popen([
                'roslaunch',
                launch_file,
                'gui:='+("true" if gui else "false"),
                                                    ])
            time.sleep(10)
            rospy.init_node('gym',anonymous=True,log_level=rospy.FATAL)
            rospy.set_param('/use_sim_time',True)
            self.gazebo_sim = GazeboSimulation(init_position=self.init_position)    #init
    
    def seed(self,seed):
        np.random.seed(seed)
    
    def reset(self):
        self.step_count = 0 #for tackle step timeout
        self.gazebo_sim.reset() #set robot by init_postion
        self.start_time = self.current_time = rospy.get_time()
        pos, psi = self._get_pos_psi()

        self.gazebo_sim.unpause()
        obs = self._get_observation()
        self.gazebo_sim.pause()

        goal_pos = np.array([self.goal_position[0] - pos.x, self.goal_position[1] - pos.y])
        self.last_goal_pos = goal_pos
        return obs
    
    def step(self,action):
        #take an action and step the Env
        self._take_action(action)
        self.step_count += 1
        pos,psi = self._get_pos_psi()

        self.gazebo_sim.unpause()
        #compute observation
        obs = self._get_observation()

        #compute termination
        flip = pos.z > 0.1
        goal_pos = np.array([self.goal_position[0] - pos.x, self.goal_position[1] - pos.y])##(goal.x-cur.x, goal.y-cur.y)
        print("goal_pos is", goal_pos)
        success = np.linalg.norm(goal_pos) < 0.5
        timeout = self.step_count >= self.max_step
        collided = self.gazebo_sim.get_collision()

        done = flip or success or timeout or collided

        #compute reward 
        rew = 0
        if done:
            if success:
                rew += self.success_reward
            if collided:
                #print("robot has collided")
                rew += self.collision_reward
            if timeout:
                #print("robot has timeout")
                rew += self.collision_reward
            if flip:
                #print("robot has fliped")
                rew += self.collision_reward
                
        
        if not done:
            #if move close to goal, get a positive rew
            rew += (np.linalg.norm(self.last_goal_pos) - np.linalg.norm(goal_pos))
            self.last_goal_pos = goal_pos

        info = dict(
            collided = collided,
            goal_position = goal_pos,
            time = self.current_time - self.start_time,
            success = success,
        )

        self.gazebo_sim.pause()
        return obs, rew, done, info
    
    def _take_action(self,action):
        linear_speed, angular_speed = action
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed

        self.gazebo_sim.unpause()
        self._cmd_vel_pub.publish(cmd_vel_value)
        # time step delay
        current_time = rospy.get_time()
        while current_time - self.current_time < self.time_step:
            time.sleep(0.01)
            current_time = rospy.get_time()
        self.current_time = current_time
        # time step delay
        self.gazebo_sim.pause()


    def _get_observation(self):
        #
        img_rgb,height,width = self.gazebo_sim.get_raw_data()
        img_rgb = cv2.resize(img_rgb, (width,height))
        img_rgb = np.array(img_rgb)

        img_rgb = img_rgb / 255.0   #for train
        return img_rgb
    
    def _get_pos_psi(self):
        pose = self.gazebo_sim.get_model_state().pose
        pos = pose.position

        q1 = pose.orientation.x
        q2 = pose.orientation.y
        q3 = pose.orientation.z
        q0 = pose.orientation.w
        psi = np.arctan2(2 * (q0*q3 + q1*q2),(1-2*(q2**2+q3**2)))##calculate yaw
        assert -np.pi <= psi <= np.pi, psi

        return pos,psi

    def reset_init_model_state(self, init_position = [0,0,0]):
        self.gazebo_sim.reset_init_model_state(init_position)

    def close(self):
        #These will make sure all the ros processed being killed
        os.system("killall -9 rosmaster")
        os.system("killall -9 gzclient")
        os.system("killall -9 gzserver")
        os.system("killall -9 roscore")