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
    import roslaunch
    from geometry_msgs.msg import Twist
    import tf  
except ModuleNotFoundError:
    pass

sys.path.append(join(os.path.dirname(__file__),'..'))
from envs.gazebo_simulation import GazeboSimulation

class TurtleGazebo(gym.Env):
    def __init__(
        self,
        gui=True,
        init_sim=True,
        init_position=[0.0, -3.0, np.pi] ,    #office: [ 0.0, -3.0, np.pi/2]   maze: [0.0, -3.0, np.pi]
        goal_position=[-3.8, -3.8, 0.0],      #office: [2.5, 3.0, np.pi/2]    maze: [-3.8, -3.8, 0.0]
        max_step=100,
        time_step=1.0,    #1s == 1 time step
        success_reward=100,
        collision_reward=-50,
        time_penalty=-0.05,
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
        self.time_penalty = time_penalty

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
            high=255,   #for collect image: 255; for train: 1
            shape=(480, 640, 3),
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
            # uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
            # roslaunch.configure_logging(uuid)
            # self.launch = roslaunch.parent.ROSLaunchParent(uuid, [launch_file])
            # self.launch.start()

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
        pos, psi, _ = self._get_pos_psi()

        self.gazebo_sim.unpause()
        obs = self._get_observation()
        self.gazebo_sim.pause()

        relative_pos = np.array([self.goal_position[0] - pos.x, self.goal_position[1] - pos.y])
        self.last_relative_pos = relative_pos
        return obs
    
    def step(self,action):
        #take an action and step the Env
        self._take_action(action)
        self.step_count += 1
        pos, psi, quat = self._get_pos_psi()

        self.gazebo_sim.unpause()
        #compute observation
        obs = self._get_observation()

        #compute camera pose
        camera_x, camera_y, camera_z, camera_orientation_x, camera_orientation_y, camera_orientation_z, camera_orientation_w = self.get_camera_pose(pos, quat)

        #compute termination
        flip = pos.z > 0.1
        relative_pos = np.array([self.goal_position[0] - pos.x, self.goal_position[1] - pos.y])##(goal.x-cur.x, goal.y-cur.y)
        #print("relative_pos is", relative_pos)
        success = np.linalg.norm(relative_pos) < 0.354
        timeout = self.step_count >= self.max_step
        collided = self.gazebo_sim.get_collision()

        done = flip or success or timeout or collided

        #compute reward 
        rew = 0
        if done:
            if success:
                rew += self.success_reward
            if collided:
                print("robot has collided")
                rew += self.collision_reward
            if timeout:
                print("robot has timeout")
            #     #rew += self.collision_reward
            # if flip:
            #     #print("robot has fliped")
            #     rew += self.collision_reward
                
        if not done:
            #if move close to goal, get a positive rew
            # move_dist  = (np.linalg.norm(self.last_relative_pos) - np.linalg.norm(relative_pos))
            # # print("move dist is: ", move_dist)
            # rew += move_dist
            rew += self.time_penalty
            self.last_relative_pos = relative_pos

        info = dict(
            collided = collided,
            relative_position = relative_pos,
            time = self.current_time - self.start_time,
            success = success,
            camera_pos = np.array([camera_x, camera_y, camera_z]),
            camera_quat = np.array([camera_orientation_x, camera_orientation_y, camera_orientation_z, camera_orientation_w]),
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
        # time.sleep(0.05)
        # time step delay
        self.gazebo_sim.pause()


    def _get_observation(self):
        #
        img_rgb,height,width = self.gazebo_sim.get_raw_data()
        img_rgb = cv2.resize(img_rgb, (width,height))
        img_rgb = np.array(img_rgb)

        #img_rgb = img_rgb / 255.0   #for train
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

        return pos, psi, pose.orientation

    def reset_init_model_state(self, init_position = [0,0,0]):
        self.gazebo_sim.reset_init_model_state(init_position)
    
    def get_camera_pose(self, robot_pos, robot_quat):
        x_offset, y_offset, z_offset = -0.087, -0.0125, 0.287
        camera_x = robot_pos.x + x_offset
        camera_y = robot_pos.y + y_offset
        camera_z = robot_pos.z + z_offset
        camera_orientation_x = robot_quat.x
        camera_orientation_y = robot_quat.y
        camera_orientation_z = robot_quat.z
        camera_orientation_w = robot_quat.w
        return camera_x, camera_y, camera_z, camera_orientation_x, camera_orientation_y, camera_orientation_z, camera_orientation_w

    def quat2rot(self, quat_):
        n = np.dot(quat_, quat_)
        if n < np.finfo(quat_.dtype).eps:
            return np.identity(3)
        quat_ = quat_ * np.sqrt(2.0 / n)
        quat_ = np.outer(quat_, quat_)
        rot_matrix = np.array(
            [
                [1.0 - quat_[2,2] - quat_[3,3], quat_[1,2] + quat_[3,0], quat_[1,3] - quat_[2,0]],
                [quat_[1,2] - quat_[3,0], 1.0 - quat_[1,1] - quat_[3,3], quat_[2,3] + quat_[1,0]],
                [quat_[1,3] + quat_[2,0], quat_[2,3] - quat_[1,0], 1.0 - quat_[1,1] - quat_[2,2]],
            ],
            dtype=quat_.type
        )

        return rot_matrix


    def close(self):
        #These will make sure all the ros processed being killed
        os.system("killall -9 rosmaster")
        os.system("killall -9 gzclient")
        os.system("killall -9 gzserver")
        os.system("killall -9 roscore")