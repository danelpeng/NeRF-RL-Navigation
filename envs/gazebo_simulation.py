import numpy as np
import os 
import cv2


try:
    import rospy
    from std_srvs.srv import Empty
    from gazebo_msgs.msg import ModelState, ContactsState
    from gazebo_msgs.srv import SetModelState,GetModelState
    from geometry_msgs.msg import Quaternion,Twist
    from sensor_msgs.msg import Image 
    from std_msgs.msg import Bool

    from cv_bridge import CvBridge
except ModuleNotFoundError:
    pass

def create_model_state(x,y,z,angle):
    #set the turtle start state
    model_state = ModelState()
    model_state.model_name = 'robot'
    model_state.pose.position.x = x
    model_state.pose.position.y = y
    model_state.pose.position.z = z
    model_state.pose.orientation = Quaternion(0,0,np.sin(angle/2.),np.cos(angle/2.))
    model_state.reference_frame = "world"

    return model_state

class GazeboSimulation():
    #
    def __init__(self,init_position = [0,0,0]):
        self._pause = rospy.ServiceProxy('/gazebo/pause_physics',Empty)
        self._unpause = rospy.ServiceProxy('/gazebo/unpause_physics',Empty)
        self._reset = rospy.ServiceProxy('/gazebo/set_model_state',SetModelState)
        self._model_state_getter = rospy.ServiceProxy('/gazebo/get_model_state',GetModelState)
        self._model_state_setter = rospy.ServiceProxy('/gazebo/set_model_state',SetModelState)
        self._init_model_state = create_model_state(x=init_position[0],y=init_position[1],z=0,angle=init_position[2])

    def get_collision(self):
        contact_data = None
        while contact_data is None:
            try:
                contact_data = rospy.wait_for_message('/contact_state', ContactsState, timeout=5)
            except:
                pass
        
        collided = contact_data.states != []
        return collided
    
    def pause(self):
        #
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self._pause()
        except rospy.ServiceException:
            print("/gazebo/pause_physics service call failed")
    
    def unpause(self):
        #
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self._unpause()
        except rospy.ServiceException:
            print("/gazebo/unpause_physics service call failed")
        
    def reset(self):
        #when a episode failed, use this to reset the new state
        rospy.wait_for_service("/gazebo/set_model_state")
        try:
            self._reset(self._init_model_state)
        except (rospy.ServiceException):
            rospy.logwarn("/gazebo/set_model_state service call failed")
    
    def get_raw_data(self):
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/camera/rgb/image_raw',Image,timeout=5)
                cv_image = CvBridge().imgmsg_to_cv2(data,"bgr8")    #for noetic
                #cv_image = self.image_to_numpy(data)    #for melodic
                h = data.height
                w = data.width
            except:
                pass
        return cv_image, h, w 
    
    def get_model_state(self):
        #
        rospy.wait_for_service("/gazebo/get_model_state")
        try:
            return self._model_state_getter('robot','world')
        except (rospy.ServiceException):
            rospy.logwarn("/gazebo/get_model_state service call failed")
    
    def set_model_state(self, model_state):
        #set robot state
        rospy.wait_for_service("/gazebo/set_model_state")
        try:
            self._model_state_setter(model_state)
        except (rospy.ServiceException):
            rospy.logwarn("gazebo/set_model_state service call failed")
    
    def reset_init_model_state(self,init_position = [0,0,0]):
        #overwrite the initial model state
        self._init_model_state = create_model_state(init_position[0],init_position[1],0,init_position[2])
    
    def image_to_numpy(self, msg):
        name_to_dtypes = {
            "bgr":(np.uint8, 3)
        }
        dtype_class, channels = name_to_dtypes[msg.encoding]
        dtype = np.dtype(dtype_class)
        dtype = dtype.newbyteorder('>' if msg.is_bigendian else '<')
        shape = (msg.height, msg.width, channels)
        data = np.fromstring(msg.data, dtype=dtype).reshape(shape)
        data.strides = (
            msg.step,
            dtype.itemsize * channels,
            dtype.itemsize
        )

        if channels == 1:
            data = data[..., 0]
        
        return data