from gym.envs.registration import register

#Motion control envs
register(
    id="motion_control_continuous_laser-v0",
    entry_point="envs.turtle_gazebo_env:TurtleGazebo"
)