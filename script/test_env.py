import gym
import numpy as np

import envs.registration

def main():
    env = gym.make(
        id = 'motion_control_continuous_laser-v0',
    )
    obs = env.reset()
    done = False
    count = 0
    ep_count = 0
    ep_rew = 0

    high = env.action_space.high 
    low = env.action_space.low
    bias = (high + low) /2
    scale = (high - low) /2
    while ep_count < 5:
        actions = 2*(np.random.rand(env.action_space.shape[0]) - 0.5)
        actions *= scale
        actions += bias

        count += 1
        obs, rew, done, info = env.step(actions)
        ep_rew += rew
        p = env.gazebo_sim.get_model_state().pose.position
        print('current episode: %d, current step: %d, time: %.2f, X position: %f(world frame),\
            Y position: %f(world frame), rew: %f'%(ep_count,count,info["time"],p.x,p.y,rew))
        print("actions: ",actions)

        if done:
            ep_count += 1
            obs = env.reset()
            count = 0
            ep_rew = 0
        
    env.close()

if __name__ == '__main__':
    main()
