from gym.envs.registration import register

"""
Jackal robot
"""
# register(
#     id="motion_control_continuous_laser-v0",
#     entry_point="envs.turtle_gazebo_env:TurtleGazebo"
# )

"""
turtlebot robot
"""
register(
    id="NeRF-RL-Nav",
    entry_point="envs.env:GazeboSimulation"
)